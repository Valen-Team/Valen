"""Validate immutable, bounded release evidence without downloading model files."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[2]


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--gate', type=Path, default=ROOT / 'validation/release-gate.json')
    p.add_argument('--evidence-dir', type=Path, help='Local website static/cookbook directory; otherwise fetch pinned small files')
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    report = {'passed':False, 'validation_kind':'release_evidence_integrity', 'model_download_bytes':0}
    try:
        gate = json.loads(args.gate.read_text())
        assert gate['completed'] is True and gate['passed'] is True
        assert len(gate['website_commit']) == 40
        fixture = hashlib.sha256((ROOT / 'tests/fixtures/backend-contract.json').read_bytes()).hexdigest()
        assert gate['fixture_sha256'] == fixture, 'Fixture changed: rerun real checks'
        for name,digest in gate['source_sha256'].items():
            assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, 'Source changed: ' + name
        files = {}
        total = 0
        for name,digest in gate['evidence_sha256'].items():
            assert '..' not in Path(name).parts and not Path(name).is_absolute()
            assert Path(name).suffix in ['.json','.py','.txt'], 'Only small evidence/source files allowed'
            if args.evidence_dir:
                raw = (args.evidence_dir / name).read_bytes()
            else:
                url = 'https://raw.githubusercontent.com/Valen-Team/valen-team.github.io/' + gate['website_commit'] + '/static/cookbook/' + name
                with urllib.request.urlopen(url, timeout=45) as response:
                    raw = response.read(512001)
            assert len(raw) <= 512000, 'Evidence unexpectedly large'
            assert hashlib.sha256(raw).hexdigest() == digest, 'Evidence hash mismatch: ' + name
            total += len(raw)
            if name.endswith('.json'):
                files[name] = json.loads(raw)
        release = files['validation/release-checks.json']
        assert release['completed'] is True and release['passed'] is True and release['fixture_sha256'] == fixture
        required = ['native-fp32','onnx-fp32','onnx-production','gguf-prebuilt','gguf-source-video']
        for backend in required:
            actual = files['validation/' + backend + '-contract.json']
            assert actual['passed'] is True and actual['fixture_sha256'] == fixture
            assert actual['records'] and all(r['passed'] is True for r in actual['records'])
        for backend in ['onnx-production','gguf-prebuilt','gguf-source-video']:
            for language in ['python','javascript']:
                actual = files['validation/' + backend + '-' + language + '-sdk.json']
                assert actual['passed'] is True and actual['sdk_version'] == '0.2.0'
                assert actual['records'] and all(r['passed'] is True for r in actual['records'])
                if language == 'javascript':
                    assert actual['sdk_name'] == '@valen-team/sdk'
        assert files['validation/cross-backend.json']['passed'] is True
        performance = files['validation/performance.json']
        assert performance['completed'] is True and performance['passed'] is False
        failures = [r for r in performance['records'] if not r['passed']]
        assert [(r['backend'],r['modality']) for r in failures] == [('gguf-q8','video')]
        assert len(performance['records']) == 9 and len([r for r in performance['records'] if r['passed']]) == 8
        assert 'video' not in files['validation/gguf-prebuilt-contract.json']['scope']
        assert 'video' in files['validation/gguf-source-video-contract.json']['scope']
        assert files['validation/preserved-failures.json']['preserved'] is True
        assert files['validation/runtime-integrity.json']['passed'] is True
        outputs = files['results.json']
        assert len(outputs['recipes']) == 5 and all(r['status'] == 'verified' for r in outputs['recipes'].values())
        for name,digest in outputs['tested_sources'].items():
            assert gate['evidence_sha256']['code/' + name] == digest
        versions = files['versions.json']
        assert versions['fixture_sha256'] == fixture
        assert versions['sdk']['python']['version'] == '0.2.0'
        assert versions['sdk']['javascript']['name'] == '@valen-team/sdk'
        assert versions['runtime']['direct_video'] is False
        report.update(passed=True, files=len(gate['evidence_sha256']), bytes_checked=total,
                      website_commit=gate['website_commit'], fixture_sha256=fixture,
                      retained_failure='prebuilt 0.1.0 direct video not compiled; excluded from positive qualification')
    except Exception as exc:
        report['error'] = str(exc)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report['passed'] else 1)


if __name__ == '__main__':
    main()
