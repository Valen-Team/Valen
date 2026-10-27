import json
import math
from types import SimpleNamespace

import pytest

from valen.reliability_collect import collect
from valen.evaluation.calibration import probabilities, summary, scored

IDENTITY = {"model": "fixture", "revision": "fixture", "backend": "cpu", "precision": "float64",
            "protocol_sha256": "a" * 64, "artifacts_sha256": {"fixture": "b" * 64}}


def records():
    return [{"group_id": "customer-1", "meta": {"record_id": "ticket-1"},
             "request": {"state": "a ticket", "questions": {"intent": {"type": "choice", "instructions": "route",
                         "criteria": {"billing": "billing", "technical": "technical"}}}},
             "targets": {"intent": {"probabilities": {"billing": 1, "technical": 0}}}}]


def test_collection_never_sends_labels_to_model_and_reorders_native_logits():
    seen = []
    def logits(request):
        seen.append(request)
        assert set(request) == {"state", "questions"}
        return {"intent": (["technical", "billing"], [0, 2])}
    model = SimpleNamespace(identity=IDENTITY, logits=logits)
    result = collect(model, records(), "test", "customers", {"name": "fixture"})
    row = result["rows"][0]
    assert row["candidates"] == ["billing", "technical"]
    assert row["logits"] == [2, 0]
    assert row["target"] == [1, 0]
    assert len(seen) == 1


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_nonfinite_logits_count_as_failed_requests(value):
    model = SimpleNamespace(identity=IDENTITY, logits=lambda request: {"intent": (["billing", "technical"], [value, 0])})
    result = collect(model, records(), "test", "customers", {"name": "fixture"})
    assert result["rows"][0]["status"] == "error"
    metrics = summary(scored(result["rows"], 1))
    assert metrics["valid_coverage"] == 0
    assert metrics["accuracy"] == 0


def test_failed_inference_retains_labeled_denominator_without_private_exception_text():
    def fail(request):
        raise RuntimeError("private input must not be retained")
    model = SimpleNamespace(identity=IDENTITY, logits=fail)
    result = collect(model, records(), "test", "customers", {"name": "fixture"})
    assert len(result["rows"]) == 1
    assert result["rows"][0]["error_type"] == "RuntimeError"
    assert "private input" not in json.dumps(result)


def test_source_group_and_unique_record_ids_required():
    model = SimpleNamespace(identity=IDENTITY, logits=lambda _: {})
    with pytest.raises(ValueError, match="Duplicate"):
        collect(model, records() * 2, "test", "customers", {})
    record = records()[0]; record.pop("group_id")
    with pytest.raises(ValueError, match="group_id"):
        collect(model, [record], "test", "customers", {})


def test_routing_policy_sends_low_probability_and_neutral_policy_to_review():
    from valen.evaluation.calibration import digest
    from valen.routing import route_choice
    artifact = {"identity": IDENTITY, "temperature": 2}
    policy = {"kind": "top_probability_policy", "identity": IDENTITY, "calibration_sha256": digest(artifact),
              "threshold": .9, "max_error": .05}
    assert route_choice(["a", "b"], [10, 0], artifact, policy)["route"] == "automatic"
    assert route_choice(["a", "b"], [0, 0], artifact, policy)["route"] == "human_review"
    policy["threshold"] = None
    assert route_choice(["a", "b"], [10, 0], artifact, policy)["route"] == "human_review"
    artifact["temperature"] = 1
    with pytest.raises(ValueError, match="different temperature"):
        route_choice(["a", "b"], [10, 0], artifact, policy)


def test_historical_replay_checks_original_log_hashes_and_release_identity(tmp_path):
    from evaluation.reliability.banking77 import collect as collect_study, file_hash
    model = "Valen-0.8B"
    data_root, release_root = tmp_path / "data", tmp_path / "release"
    run = data_root / "results/full" / model / "Banking77"
    run.mkdir(parents=True)
    (data_root / "datasets").mkdir()
    source_path = data_root / "datasets/Banking77.tsv"
    source_path.write_text("software-fixture-only")
    runtime = {"model": model, "revision": "main", "execution": "shared_state", "attention": "sdpa",
               "dtype": "torch.float32", "temperature": 1.0, "config_sha256": "b" * 64,
               "weights": {"weights.safetensors": "a" * 64}, "runtime_sha256": {"native.py": "c" * 64}}
    manifest = {"signature": {"limit": None, "dataset_sha256": file_hash(source_path),
                               "source_sha256": {"adapter.py": "d" * 64}}, "runtime": runtime}
    (run / "manifest.json").write_text(json.dumps(manifest))
    keys = [f"c{i}" for i in range(77)]
    question = {"type": "choice", "instructions": "route", "criteria": dict.fromkeys(keys, "fixture description")}
    source = [{"index": str(i), "request": {"state": f"fixture ticket {i}", "questions": {"q": question}},
               "targets": {"q": "c0"}} for i in range(100)]
    p = dict.fromkeys(keys, .2 / 76); p["c0"] = .8
    predictions = [{"index": str(i), "status": "ok", "response": {"answers": {"q": {"probabilities": p}}}}
                   for i in range(100)]
    (run / "predictions.jsonl").write_text("\n".join(json.dumps(r) for r in predictions))
    (run / "metrics.json").write_text(json.dumps({"predictions_sha256": file_hash(run / "predictions.jsonl"),
                                                 "manifest_sha256": file_hash(run / "manifest.json")}))
    release = release_root / "runs" / model
    release.mkdir(parents=True)
    published = {"model_source": {"repo_id": "Valen-Team/" + model, "revision": "pinned-fixture",
                                  "inference_files_sha256": {"config.json": "b" * 64,
                                                            "weights.safetensors": "a" * 64,
                                                            "native.py": "c" * 64}}}
    (release / "environment.json").write_text(json.dumps(published))
    caches, _ = collect_study(data_root, model, source, release_root)
    assert sum(len(c["rows"]) for c in caches.values()) == 100
    assert all(c["identity"]["revision"] == "pinned-fixture" for c in caches.values())
    group_sets = [{r["group_id"] for r in c["rows"]} for c in caches.values()]
    assert not group_sets[0] & group_sets[1] and not group_sets[1] & group_sets[2]
    with (run / "predictions.jsonl").open("a") as stream: stream.write("\n")
    with pytest.raises(ValueError, match="integrity mismatch"):
        collect_study(data_root, model, source, release_root)
