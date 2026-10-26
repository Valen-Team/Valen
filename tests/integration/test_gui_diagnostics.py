"""Diagnostic cohorts and coordinate parsing remain independent of target labels."""
import copy

import pytest

from evaluation.gui.diagnose import compare_reports, hit, parse_point, select


@pytest.mark.parametrize("response, expected", [
    ('<tool_call>{"name":"computer_use","arguments":{"coordinate":[100,200]}}</tool_call>', [.1, .2]),
    ('[100, 200, 300, 400]', [.2, .3]),
    ('[0, 1000]', [0., 1.]),
    ('[false, 200]', None),
    ('[1001, 200]', None),
    ('[300, 200, 100, 400]', None),
    ('[NaN, 200]', None),
    ('no prediction', None),
])
def test_native_coordinates_use_one_fixed_rule(response, expected):
    assert parse_point(response) == expected


def test_diagnostic_selection_and_resolution_cohorts_do_not_use_targets():
    rows = [dict(id=str(i), meta=dict(ui_type="icon" if i % 2 else "text"), targets=dict(box=[i])) for i in range(20)]
    expected = [r["id"] for r in select(rows, 3)]
    changed = copy.deepcopy(list(reversed(rows)))
    for r in changed:
        r["targets"] = dict(box=[999, -1])
    assert [r["id"] for r in select(changed, 3)] == expected
    assert len(expected) == len(set(expected)) == 6
    with pytest.raises(ValueError, match="Insufficient"):
        select(rows, 11)


def test_point_scoring_matches_inclusive_official_rule():
    box = [.1, .2, .3, .4]
    assert hit([.1, .2], box) and hit([.3, .4], box)
    assert not hit([.09, .3], box) and not hit(None, box)


def test_paired_reports_reject_changed_data_but_expose_geometry_differences():
    cases = [dict(id=b, benchmark=b, max_pixels=100, data_sha256="same", image_grid_thw=[1, 4, 4],
                  point=[.2, .3], hit=True) for b in ("screenspot_pro_test", "screenspot_v2_test")]
    reference = dict(model="spatial", mode="spatial", world_size=8, gpu_names=["H200"] * 8, max_pixels=[100], cases=cases)
    native = copy.deepcopy(dict(reference, model="native", mode="native"))
    native["cases"][0]["image_grid_thw"] = [1, 8, 8]
    summary = compare_reports([reference, native])
    assert summary["models"] == 2 and summary["cases_per_model"] == 2
    assert sum(row["image_grid_mismatches"] for row in summary["table"]) == 1
    native["cases"][0]["data_sha256"] = "changed"
    with pytest.raises(ValueError, match="identities"):
        compare_reports([reference, native])


def test_progress_requires_a_matching_data_audit_and_complete_diagnostic_scope(tmp_path):
    import json
    from scripts.train.spatial_progress import write_progress
    data = tmp_path / "data"
    manifest = data / "eval/gui/manifests/screenspot_pro_test.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps(dict(samples=1581, records_sha256="current")))
    audit = tmp_path / "gui-pro-diagnosis/data-audit.json"
    audit.parent.mkdir()
    payload = dict(records=1581, records_sha256="old", media_match_official_bytes=True,
                   official_gold_center_accuracy=1, local_official_point_scorer_agreement=1)
    audit.write_text(json.dumps(payload))
    diagnostic = tmp_path / "diagnostic"
    diagnostic.mkdir()
    state = dict(jobs=[], gui_diagnosis=dict(run_dir=str(diagnostic), expected_models=8, cases_per_model=128))
    output = tmp_path / "progress.md"
    def render():
        write_progress(state, tmp_path / "registry.json", data, tmp_path / "models", output)
        return output.read_text()
    assert "- [x] 已完成：全部官方标注" not in render()
    payload["records_sha256"] = "current"
    audit.write_text(json.dumps(payload))
    assert "- [x] 已完成：全部官方标注" in render()
    marker = dict(models=8, cases_per_model=128)
    (diagnostic / "pipeline_complete.json").write_text(json.dumps(marker))
    (diagnostic / "paired_summary.json").write_text(json.dumps(dict(marker, models=2, table=[{}])))
    assert "- [x] 已完成：原生基座" not in render()
    table = [dict(model=("Qwen3.5-" if mode == "native" else "Valen-Decision-GUI-") + size,
                  mode=mode, benchmark=benchmark, max_pixels=pixels, samples=32)
             for size in ("0.8B", "2B", "4B", "9B") for mode in ("native", "spatial")
             for benchmark in ("screenspot_pro_test", "screenspot_v2_test") for pixels in (1048576, 4194304)]
    (diagnostic / "paired_summary.json").write_text(json.dumps(dict(marker, table=table)))
    assert "- [x] 已完成：原生基座" in render()
