"""All requested LoRA evaluations must precede full-training comparisons."""
import json

from scripts.train.schedule_spatial import EVALUATED_CHECKS, publish_evaluation_barriers


def test_three_model_barrier_rejects_partial_reasonseg_and_wrong_identity(tmp_path):
    names = ["Valen-Decision-Det-" + size for size in ("0.8B", "2B", "4B")]
    marker = tmp_path / "evaluated.json"
    state = {"evaluation_barriers": [{"models": names, "marker": str(marker)}]}
    report = {"models": [{"name": name, "variant": "lora", "export_identity": {"source_checkpoint_sha256": name},
        "checks": dict.fromkeys(EVALUATED_CHECKS, True)} for name in names]}
    for missing in ("required_benchmarks", "reasonseg", "evaluation_model_identity"):
        report["models"][2]["checks"][missing] = False
        publish_evaluation_barriers(state, report)
        assert not marker.exists() and not state["evaluation_barriers"][0]["ready"]
        report["models"][2]["checks"][missing] = True
    publish_evaluation_barriers(state, report)
    assert marker.is_file() and state["evaluation_barriers"][0]["ready"]
    assert json.loads(marker.read_text())["models"] == names
    report["models"][0]["checks"]["joint_epoch"] = False
    publish_evaluation_barriers(state, report)
    assert not marker.exists()


def test_new_supplement_cannot_be_skipped_by_a_preexisting_pipeline_marker(tmp_path):
    name = "Valen-Decision-Det-4B"
    marker = tmp_path / "evaluated.json"
    state = {"evaluation_barriers": [{"models": [name], "marker": str(marker),
        "required_checks": ["groundingme", "groundingme_model_identity"]}]}
    model = {"name": name, "variant": "lora", "export_identity": {},
             "checks": dict.fromkeys(EVALUATED_CHECKS, True)}
    report = {"models": [model]}
    publish_evaluation_barriers(state, report)
    assert not marker.exists()
    model["checks"].update(groundingme=True, groundingme_model_identity=False)
    publish_evaluation_barriers(state, report)
    assert not marker.exists()
    model["checks"]["groundingme_model_identity"] = True
    publish_evaluation_barriers(state, report)
    assert marker.is_file()
    assert "groundingme" in json.loads(marker.read_text())["verified_checks"]


def test_full_vs_lora_gate_requires_full_comparison_and_both_complete_suites(tmp_path):
    lora, full = "Valen-Decision-Det-2B", "Valen-Decision-Det-2B-Full"
    marker = tmp_path/"comparison-ready.json"
    common = ["groundingme", "groundingme_model_identity", "inference_examples"]
    extra = ["full_comparison", "reasonseg_comparison_data", "groundingme_comparison_data"]
    barrier = dict(models=[lora,full],marker=str(marker),expected_variants={full:"full"},
                   required_checks=common,model_required_checks={full:extra})
    state = {"evaluation_barriers":[barrier]}
    report = {"models":[dict(name=name,variant=variant,export_identity={},
        checks=dict.fromkeys([*EVALUATED_CHECKS,*common,*extra],True))
        for name,variant in [(lora,"lora"),(full,"full")]]}
    for check in extra+common:
        report["models"][1]["checks"][check]=False
        publish_evaluation_barriers(state,report)
        assert not marker.exists()
        report["models"][1]["checks"][check]=True
    publish_evaluation_barriers(state,report)
    assert marker.exists() and barrier["ready"]
    assert json.loads(marker.read_text())["format"]=="valen_evaluated_comparison_barrier_v1"
    report["models"][0]["checks"]["required_benchmarks"]=False
    publish_evaluation_barriers(state,report)
    assert not marker.exists()


def test_serial_rlcd_gate_waits_for_post_replay_and_all_benchmarks(tmp_path):
    name = "Valen-Decision-Det-2B-RLCD"
    marker = tmp_path / "rlcd-ready.json"
    extra = ["groundingme", "groundingme_model_identity", "inference_examples",
             "full_reference_identity", "rlcd_comparison", "rlcd_resume_verified",
             "training_error_comparison"]
    barrier = dict(models=[name], marker=str(marker), expected_variants={name: "rlcd"},
                   required_checks=extra, require_all_deliverables=True, allow_zero_error_skip=True)
    model = dict(name=name, variant="rlcd", export_identity={}, deliverables_ready=True,
                 checks=dict.fromkeys([*EVALUATED_CHECKS, *extra], True))
    state, report = {"evaluation_barriers": [barrier]}, {"models": [model]}
    for key in extra:
        model["checks"][key] = False
        publish_evaluation_barriers(state, report)
        assert not marker.exists()
        model["checks"][key] = True
    model["deliverables_ready"] = False
    publish_evaluation_barriers(state, report)
    assert not marker.exists()
    model["deliverables_ready"] = True
    publish_evaluation_barriers(state, report)
    assert json.loads(marker.read_text())["format"] == "valen_evaluated_rlcd_barrier_v1"
    # A zero-error predecessor can release the next size only with its audited skip.
    model.update(skipped=True, checks={"audited_zero_error_skip": False})
    publish_evaluation_barriers(state, report)
    assert not marker.exists()
    model["checks"]["audited_zero_error_skip"] = True
    publish_evaluation_barriers(state, report)
    assert json.loads(marker.read_text())["audited_zero_error_skips"] == [name]
