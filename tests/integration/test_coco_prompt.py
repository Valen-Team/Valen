"""Conditional prompts retain every requested instance and cap training image reuse."""
from collections import Counter
import copy
import json
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from evaluation.det.audit import gold_audit
from evaluation.spatial import evaluate
from scripts.data.prepare_coco_prompt import development_in_eval, grouped_queries, training_queries
from scripts.data.prepare_spatial import digest_file, make_record, save_dataset, write_json
from scripts.train.run_spatial_pipeline import required_benchmarks
from scripts.train.schedule_spatial import data_ready
from valen.modeling.qwen.spatial import SpatialOutput
from valen.training.distributed import Distributed


def example(identifier=7):
    labels = [0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    return make_record("det", f"coco:{identifier}", f"coco:{identifier}", "image.jpg", "Detect.",
        [f"class-{i}" for i in range(10)],
        [[.02 * i, .02 * i, .02 * i + .1, .02 * i + .1] for i in range(len(labels))], labels,
        source="coco2017", image_id=identifier, width=320, height=160,
        category_ids=[1, 3, 5, 7, 9, 11, 18, 22, 88, 90])


def test_category_groups_preserve_all_instances_and_sparse_ids():
    source = example()
    unchanged = copy.deepcopy(source)
    groups = list(grouped_queries(source))
    assert groups == list(grouped_queries(source)) and source == unchanged
    recovered = []
    for group in groups:
        question, target = group["request"]["questions"]["target"], group["targets"]["target"]
        assert len(question["categories"]) == 1
        assert set(target["labels"]) == set(range(len(question["categories"])))
        assert all(name in question["instructions"] for name in question["categories"])
        recovered.extend((tuple(box), group["meta"]["category_ids"][label])
                         for box, label in zip(target["boxes"], target["labels"]))
    expected = [(tuple(box), source["meta"]["category_ids"][label])
                for box, label in zip(source["targets"]["target"]["boxes"], source["targets"]["target"]["labels"])]
    assert Counter(recovered) == Counter(expected)
    assert len(groups) == 10
    with pytest.raises(ValueError, match="exactly one"):
        list(grouped_queries(source, max_categories=80))


def test_training_uses_each_image_once_with_one_category_and_all_its_instances():
    records = [example(i) for i in range(20)]
    primary, extras = training_queries(records)
    assert len(primary) == len(records) and extras == []
    assert max(Counter(r["group_id"] for r in primary).values()) == 1
    for original, selected in zip(records, primary):
        category = selected["meta"]["category_ids"][0]
        expected = [box for box, label in zip(original["targets"]["target"]["boxes"],
                    original["targets"]["target"]["labels"]) if original["meta"]["category_ids"][label] == category]
        assert selected["targets"]["target"]["boxes"] == expected
        assert set(selected["targets"]["target"]["labels"]) == {0}
    with pytest.raises(ValueError, match="each image exactly once"):
        training_queries([records[0], records[0]])
    empty = example()
    empty["targets"]["target"] = {"boxes": [], "labels": []}
    assert list(grouped_queries(empty)) == []


def test_single_category_seal_routes_to_its_development_and_evaluation_files(tmp_path):
    det = tmp_path / "det"
    seal = {"format": "valen_coco_single_category_data_v3", "validated": True,
            "protocol": "coco_single_category_v3", "max_categories": 1, "max_coco_prompts_per_image": 1,
            "global_evaluation_overlap": 0, "independent_media": True, "records_sha256": "train"}
    write_json(det / "manifests/ready.json", seal)
    write_json(det / "manifests/train.json", {"samples": 100000, "records_sha256": "train"})
    (det / "processed").mkdir()
    (det / "processed/train.jsonl").write_text("{}")
    evaluation = tmp_path / "eval/det"
    for name in required_benchmarks("det", "conditional") | {"dev"}:
        base = evaluation / "protocols/coco_prompt" if name.startswith("coco2017_") or name == "dev" else evaluation
        path = base / "processed" / (name + ".jsonl")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")
    assert data_ready(tmp_path, "det")
    seal["max_categories"] = 3
    write_json(det / "manifests/ready.json", seal)
    assert not data_ready(tmp_path, "det")


def legacy_development(root):
    training = root / "det"
    training.mkdir()
    Image.new("RGB", (320, 160)).save(training / "image.jpg")
    records = list(grouped_queries(example()))
    manifest = save_dataset(training, "dev_coco_prompt_v2", records, ["fixture"],
                            {"image.jpg": digest_file(training / "image.jpg")})
    write_json(training / "manifests/ready.json", {"format": "valen_coco_prompt_data_v2",
        "validated": True, "development_sha256": manifest["records_sha256"]})
    return training, manifest


def test_development_relocation_preserves_live_inputs_and_survives_their_cleanup(tmp_path):
    training, original = legacy_development(tmp_path)
    legacy = training / "processed/dev_coco_prompt_v2.jsonl"
    before = legacy.read_bytes()
    migrated = development_in_eval(tmp_path)
    canonical = tmp_path / "eval/det/protocols/coco_prompt/processed/dev.jsonl"
    media = tmp_path / "eval/det/image.jpg"
    assert canonical.read_bytes() == before == legacy.read_bytes()
    assert migrated["records_sha256"] == original["records_sha256"]
    assert migrated["purpose"] == "development" and migrated["independent_media"]
    assert media.stat().st_nlink == 1 and media.stat().st_ino != (training / "image.jpg").stat().st_ino
    (training / "image.jpg").unlink()
    legacy.unlink()
    assert development_in_eval(tmp_path)["records_sha256"] == original["records_sha256"]
    media.write_bytes(b"changed")
    with pytest.raises(ValueError, match="independent verified"):
        development_in_eval(tmp_path)


def test_development_relocation_rejects_shared_media(tmp_path):
    training, _ = legacy_development(tmp_path)
    media = tmp_path / "eval/det/image.jpg"
    media.parent.mkdir(parents=True)
    media.symlink_to(training / "image.jpg")
    with pytest.raises(ValueError, match="independent verified"):
        development_in_eval(tmp_path)


def test_grouped_perfect_predictions_score_all_instances_across_prompts(tmp_path):
    source = example()
    questions = []
    for record in grouped_queries(source):
        target = record["targets"]["target"]
        names = record["request"]["questions"]["target"]["categories"]
        logits = torch.full((len(target["boxes"]), len(names) + 1), -20.)
        for row, label in enumerate(target["labels"]):
            logits[row, label] = 20
        questions.append(SimpleNamespace(kind="det", original_size=(160, 320), categories=names,
            metadata=record["meta"], target=target,
            output=SpatialOutput(logits, boxes=torch.tensor(target["boxes"]))))
    annotations = tmp_path / "annotations/instances_val2017.json"
    annotations.parent.mkdir()
    annotations.write_text(json.dumps({"info": {},
        "images": [{"id": i, "width": 320, "height": 160} for i in (7, 8)],
        "categories": [{"id": value, "name": name} for value, name in
            zip(source["meta"]["category_ids"], source["request"]["questions"]["target"]["categories"])],
        "annotations": [{"id": i + 1, "image_id": 7, "category_id": source["meta"]["category_ids"][label],
            "bbox": [box[0] * 320, box[1] * 160, (box[2] - box[0]) * 320, (box[3] - box[1]) * 160],
            "area": (box[2] - box[0]) * 320 * (box[3] - box[1]) * 160, "iscrowd": 0}
            for i, (box, label) in enumerate(zip(source["targets"]["target"]["boxes"], source["targets"]["target"]["labels"]))]}))
    compiler = SimpleNamespace(compile=lambda record: SimpleNamespace(questions=[record]))
    result = evaluate(lambda q: q.output, compiler, questions, "det", Distributed(), annotations)
    assert result["AP"] == pytest.approx(1) and result["AP50"] == pytest.approx(1)
    assert "coco2017_conditional_val2017" in required_benchmarks("det", "conditional")
    assert "coco2017_val2017" not in required_benchmarks("det", "conditional")
    Image.new("RGB", (320, 160)).save(tmp_path / "image.jpg")
    empty_image = tmp_path / "images/coco/000000000008.jpg"
    empty_image.parent.mkdir(parents=True)
    Image.new("RGB", (320, 160)).save(empty_image)
    prepared = tmp_path / "protocols/coco_prompt/processed/coco2017_conditional_val2017.jsonl"
    prepared.parent.mkdir(parents=True)
    groups = list(grouped_queries(source))
    prepared.write_text("".join(json.dumps(record) + "\n" for record in groups))
    report = gold_audit(tmp_path, tmp_path / "audit")
    assert report["images"] == 2 and report["queried_images"] == 1 and report["prompts"] == 10
    assert report["perfect_prediction_metrics"]["AP"] == pytest.approx(1)
    prepared.write_text("".join(json.dumps(record) + "\n" for record in groups[:-1]))
    with pytest.raises(ValueError, match="cover every"):
        gold_audit(tmp_path, tmp_path / "incomplete-audit")
