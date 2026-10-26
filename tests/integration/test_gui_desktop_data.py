"""GUI revision filters preserve supervision and prevent screenshot leakage."""
from collections import Counter
import io
import random

import numpy as np
from PIL import Image
import pytest

from scripts.data.prepare_gui_desktop import clean_elements, choose_images, image_features, NearIndex


def test_ambiguous_query_is_rejected_instead_of_choosing_an_arbitrary_target():
    counts = Counter()
    original = "  Open Project  "
    rows = [(original, [.1, .1, .2, .2], "button"),
            ("open   project", [.5, .5, .6, .6], "button"),
            ("Save File", [.2, .2, .3, .3], "button")]
    assert clean_elements(rows, (100, 100), counts) == [rows[-1]]
    assert counts["ambiguous_instruction"] == 2
    result = clean_elements([rows[0]], (100, 100), Counter())
    assert result[0][0] == original.strip()
    assert result[0][1] == rows[0][1]


def test_pointer_descriptions_are_removed_without_removing_real_cursor_commands():
    elements=[(name,[.1,.1,.2,.2],"button") for name in
              ("cursor", "cursor movement", "cursor position unchanged", "3D cursor", "Add Cursor Above")]
    for i,item in enumerate(elements):
        elements[i]=(item[0],[.1+i/10,.1,.15+i/10,.2],item[2])
    counts=Counter()
    result=clean_elements(elements,(1000,1000),counts,desktop=True,exclude_pointer_labels=True)
    assert {item[0] for item in result}=={"3D cursor","Add Cursor Above"}
    assert counts['pointer_annotation']==3


def test_bad_boxes_are_not_clipped_or_fabricated_and_target_aliases_count_once():
    counts = Counter()
    rows = [("Save File", [-.1, .1, .2, .2], "button"),
            ("Open File", [0, 0, float("nan"), .2], "button"),
            ("Close Window", [0, 0, True, .2], "button"),
            ("Menu", [.1, .1, .2, .2], "menu"),
            ("Zoom In", [.1, .1, .2, .2], "button"),
            ("Increase Zoom", [.1, .1, .2, .2], "button")]
    result = clean_elements(rows, (100, 100), counts)
    assert len(result) == 1
    assert counts["invalid_box"] == 3
    assert counts["generic_text"] == 1
    assert counts["duplicate_target_alias"] == 1


def test_reencoded_identical_screenshot_retains_pixel_identity():
    image = Image.new("RGB", (50, 30), (32, 20, 10))
    first, second = io.BytesIO(), io.BytesIO()
    image.save(first, format="PNG", compress_level=0)
    image.save(second, format="PNG", compress_level=9)
    a, b = image_features(first.getvalue()), image_features(second.getvalue())
    assert a["raw"] != b["raw"] and a["pixels"] == b["pixels"]
    index = NearIndex()
    index.add(a)
    assert index.matches(b) == [0]


def test_feature_cache_is_verified_and_corrupt_entries_are_recomputed(tmp_path):
    stream = io.BytesIO()
    Image.new("RGB", (50, 30), (32, 20, 10)).save(stream, format="PNG")
    data = stream.getvalue()
    first = image_features(data, tmp_path)
    cached = next(tmp_path.glob("*/*.json"))
    cached.write_text('{"corrupt":true}')
    second = image_features(data, tmp_path)
    third = image_features(data, tmp_path)
    for feature in (second, third):
        assert feature["pixels"] == first["pixels"]
        assert feature["dhash"] == first["dhash"]
        assert np.array_equal(feature["thumbnail"], first["thumbnail"])


def test_truncated_media_cannot_produce_a_valid_fingerprint(tmp_path):
    stream = io.BytesIO()
    Image.new("RGB", (100, 100), (32, 20, 10)).save(stream, format="PNG")
    with pytest.raises(OSError):
        image_features(stream.getvalue()[:50], tmp_path)
    assert not list(tmp_path.rglob("*.json"))


def test_near_duplicate_index_has_no_false_negative_within_eight_hash_bits():
    base = dict(pixels="a", size=(100, 100), dhash=0, thumbnail=np.zeros((36, 64, 3), dtype=np.uint8))
    index = NearIndex()
    index.add(base)
    rng = random.Random(42)
    for _ in range(100):
        positions = rng.sample(range(128), 8)
        other = dict(base, pixels="b", dhash=sum(1 << i for i in positions))
        assert index.matches(other) == [0]
    assert index.matches(dict(base, pixels="b", thumbnail=np.full((36, 64, 3), 255, dtype=np.uint8))) == []


def test_sampling_covers_images_before_repeating_them_and_fails_when_short():
    groups = [dict(application=str(i % 2), feature=dict(size=(100, 100)),
                   elements=[(f"target {j}", [.1, .1, .2, .2], "button") for j in range(3)]) for i in range(6)]
    selected = choose_images(groups, 8, random.Random(42))
    assert len({id(group) for group, _ in selected[:6]}) == 6
    assert len(selected) == 8
    with pytest.raises(ValueError, match="need 19"):
        choose_images(groups, 19, random.Random(42))


def test_expansion_preserves_existing_targets_and_rejects_reviewed_errors():
    from scripts.data.prepare_gui_desktop import element_id
    elements = [(f"target {i}", [i / 100, .1, (i + .5) / 100, .2], "button") for i in range(20)]
    group = dict(source="desktop", application="app", feature=dict(pixels="image", size=(1000, 1000)),
                 elements=elements)
    retained = {"image": [dict(request=dict(questions=dict(target=dict(instructions=item[0]))),
                                targets=dict(target=dict(boxes=[item[1]]))) for item in elements[:10]]}
    rejected = {element_id(group, elements[10])}
    selected = choose_images([group], 2, random.Random(42), retained=retained, rejected=rejected)
    assert len(selected) == 2
    assert all(element_id(g, item) not in rejected and item not in elements[:10] for g, item in selected)
    with pytest.raises(ValueError, match="need 3"):
        choose_images([group], 3, random.Random(42), retained=retained, rejected=rejected)


def test_expanded_gui_readiness_requires_matching_count_and_complete_benchmarks(tmp_path):
    import json
    from scripts.train.schedule_spatial import data_ready
    from scripts.train.run_spatial_pipeline import required_benchmarks
    manifests = tmp_path / "gui/manifests"
    manifests.mkdir(parents=True)
    (manifests / "ready.json").write_text(json.dumps(dict(format="valen_spatial_data_v1", validated=True,
        samples=172000, records_sha256="expanded", global_evaluation_overlap=0, independent_media=True)))
    manifest = manifests / "train.json"
    manifest.write_text(json.dumps(dict(samples=100000, records_sha256="expanded")))
    train = tmp_path / "gui/processed/train.jsonl"
    train.parent.mkdir()
    train.touch()
    evaluation = tmp_path / "eval/gui/processed"
    evaluation.mkdir(parents=True)
    for name in required_benchmarks("gui") | {"gui_dev"}:
        (evaluation / (name + ".jsonl")).touch()
    assert not data_ready(tmp_path, "gui")
    manifest.write_text(json.dumps(dict(samples=172000, records_sha256="expanded")))
    assert data_ready(tmp_path, "gui")
    (evaluation / "screenspot_pro_test.jsonl").unlink()
    assert not data_ready(tmp_path, "gui")


@pytest.mark.parametrize("verification_fails", [False, True])
def test_gui_install_swaps_both_directories_or_restores_both(tmp_path, monkeypatch, verification_fails):
    import json
    from scripts.data import install_gui_revision as module
    source, destination = tmp_path / "revision", tmp_path / "data"
    for root, marker in ((source, "new"), (destination, "old")):
        for relative in ("gui", "eval/gui"):
            directory = root / relative
            directory.mkdir(parents=True)
            (directory / "marker").write_text(marker)
        (root / "gui/manifests").mkdir()
        for name in ("ready.json", "sources.json"):
            (root / "gui/manifests" / name).write_text(json.dumps(dict(version=marker)))
    det = destination / "det/processed/train.jsonl"
    det.parent.mkdir(parents=True)
    det.write_text("unchanged Det data")
    guard = {"det/processed/train.jsonl": module.digest_file(det)}
    seal = dict(samples=100000, records_sha256="new-hash", revision="test")
    monkeypatch.setattr(module, "preflight", lambda *args: (seal, guard))
    def verify(root):
        if root == destination and verification_fails:
            raise ValueError("simulated installed verification failure")
        return seal
    monkeypatch.setattr(module, "verify", verify)
    if verification_fails:
        with pytest.raises(ValueError, match="simulated installed"):
            module.install(source, destination, tmp_path / "registry", tmp_path / "report.json", apply=True)
    else:
        report = module.install(source, destination, tmp_path / "registry", tmp_path / "report.json", apply=True)
        assert report["applied"] and report["retired_gui_bytes_removed"] > 0
    expected = "old" if verification_fails else "new"
    assert (destination / "gui/marker").read_text() == expected
    assert (destination / "eval/gui/marker").read_text() == expected
    assert det.read_text() == "unchanged Det data"


def test_install_preflight_rejects_active_gui_training_before_reading_data(tmp_path):
    import json
    from scripts.data.install_gui_revision import preflight
    registry = tmp_path / "registry.json"
    registry.write_text(json.dumps(dict(jobs=[dict(name="gui-job", task="gui", state="running")])))
    with pytest.raises(ValueError, match="Stop active GUI"):
        preflight(tmp_path / "revision", tmp_path / "data", registry)
