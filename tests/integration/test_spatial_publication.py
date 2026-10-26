"""Publication keeps merged weights intact and excludes checkout metadata."""
import json

import pytest

from scripts.export.qwen.publish_spatial import digest, install, merged_manifest, model_card, BENCHMARKS


def test_install_materializes_weights_and_preserves_git_metadata(tmp_path):
    original = tmp_path / 'original.safetensors'
    original.write_bytes(b'immutable merged weights')
    source, target = tmp_path / 'release', tmp_path / 'checkout'
    source.mkdir()
    (source / 'model.safetensors').symlink_to(original)
    (source / 'README.md').write_text('new model card')
    (source / 'training.log').write_text('not a release file')
    (target / '.git').mkdir(parents=True)
    (target / '.git/config').write_text('existing remote')
    (target / 'README.md').write_text('placeholder')
    files = {name: dict(sha256=digest(source/name), bytes=(source/name).stat().st_size)
             for name in ['model.safetensors', 'README.md']}
    install(source, target, files)
    assert not (target / 'model.safetensors').is_symlink()
    assert (target / 'model.safetensors').stat().st_nlink == 1
    assert (target / 'model.safetensors').read_bytes() == original.read_bytes()
    assert (target / '.git/config').read_text() == 'existing remote'
    assert not (target / 'training.log').exists()
    before = (target / 'model.safetensors').stat().st_mtime_ns
    install(source, target, files)
    assert (target / 'model.safetensors').stat().st_mtime_ns == before


def test_unvalidated_and_full_exports_cannot_be_published_as_merged_lora(tmp_path):
    manifest = dict(model_name='Valen-Decision-Det-2B', finetuning_type='full',
                    reload_validation=dict(independent_process=True))
    (tmp_path/'export_manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='merged SFT LoRA'):
        merged_manifest(tmp_path)
    manifest['finetuning_type'] = 'lora'
    manifest['reload_validation']['independent_process'] = False
    (tmp_path/'export_manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='merged SFT LoRA'):
        merged_manifest(tmp_path)


def test_model_card_has_all_measured_scores_and_executable_example():
    import ast
    benchmarks = {key: dict(accuracy=.1234) for key, _ in BENCHMARKS}
    card = model_card('Valen-Decision-Det-2B','Qwen/Qwen3.5-2B',
                      'Valen-Team/Valen-Decision-Det-2B',benchmarks)
    assert card.count('| 12.34 |') == 10
    assert 'GroundingME' not in card
    code = card.split('```python\n',1)[1].split('```',1)[0]
    ast.parse(code)
    assert 'snapshot_download("Valen-Team/Valen-Decision-Det-2B")' in code
    assert 'trust_remote_code=True' in code and 'dtype="auto"' in code
    assert '/mnt/' not in card and '/home/' not in card
