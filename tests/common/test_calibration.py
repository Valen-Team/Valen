import copy
import json
import math
import tempfile
import unittest
from pathlib import Path

from valen.evaluation.calibration import (
    audit_report, digest, fit, load_cache, load_calibration, main, probabilities, report,
    risk_curve, scored, select_policy, split_cache, summary, verify_runtime_identity, write_json,
)


def cache(role="calibration", count=100):
    return {"schema_version": 1, "role": role,
            "identity": {"model": "unit-test-model", "revision": "test-revision", "backend": "cpu-test",
                         "precision": "float64", "protocol_sha256": "a" * 64,
                         "artifacts_sha256": {"test-weights": "b" * 64}},
            "dataset": {"name": "synthetic software fixture, not model evidence"},
            "group_namespace": "test-entities",
            "rows": [{"id": f"{role}-{i}", "group_id": f"{role}-{i}", "type": "choice",
                      "candidates": ["a", "b"], "target": [1, 0] if i % 5 else [0, 1],
                      "status": "ok", "logits": [math.log(99), 0]} for i in range(count)]}


class ReliabilityTests(unittest.TestCase):
    def test_temperature_finds_known_nll_minimum(self):
        c = cache()
        artifact = fit(c)
        self.assertAlmostEqual(artifact["temperature"], math.log(99) / math.log(4), places=8)
        before = summary(scored(c["rows"], 1))
        after = summary(scored(c["rows"], artifact["temperature"]))
        self.assertLess(after["nll"], before["nll"])
        self.assertAlmostEqual(after["ece"], 0, places=10)
        self.assertEqual(before["accuracy"], after["accuracy"])

    def test_soft_targets_are_not_fabricated_hard_labels(self):
        c = cache(count=1)
        c["rows"][0]["target"] = [.8, .2]
        item = summary(scored(c["rows"], 1))
        self.assertIsNone(item["accuracy"])
        self.assertIsNone(item["ece"])
        self.assertEqual(item["nll_questions"], 1)
        c["rows"][0]["gold"] = "a"
        self.assertEqual(summary(scored(c["rows"], 1))["accuracy"], 1)

    def test_failed_requests_remain_in_coverage_and_accuracy(self):
        c = cache(count=2)
        c["rows"][0]["status"] = "unsupported"
        curve = risk_curve(scored(c["rows"], 1))
        self.assertEqual(curve[-1]["coverage"], .5)
        self.assertEqual(summary(scored(c["rows"], 1))["accuracy"], .5)

    def test_ties_are_not_split_to_cherry_pick_coverage(self):
        points = risk_curve(scored(cache(count=10)["rows"], 1))
        self.assertEqual(len(points), 2)
        self.assertEqual(points[-1]["accepted"], 10)

    def test_policy_is_fixed_before_test(self):
        artifact = fit(cache())
        policy = select_policy(cache("policy"), artifact, .95, 30)
        result = report(cache("test"), artifact, policy)
        self.assertEqual(result["operating_point"]["threshold"], policy["threshold"])
        self.assertEqual(result["operating_point"]["accepted"], 100)
        self.assertAlmostEqual(result["operating_point"]["error_rate"], .2)

    def test_no_safe_policy_accepts_nothing(self):
        artifact = fit(cache())
        policy = select_policy(cache("policy"), artifact, .01)
        self.assertIsNone(policy["threshold"])
        result = report(cache("test"), artifact, policy)
        self.assertEqual(result["operating_point"]["accepted"], 0)
        self.assertIsNone(result["operating_point"]["error_rate"])

    def test_source_group_overlap_is_rejected_even_with_different_row_ids(self):
        artifact = fit(cache())
        c = cache("test")
        c["rows"][0]["group_id"] = "calibration-0"
        with self.assertRaisesRegex(ValueError, "overlap"):
            report(c, artifact)

    def test_policy_test_overlap_is_rejected(self):
        artifact = fit(cache())
        policy = select_policy(cache("policy"), artifact, .95)
        c = cache("test")
        c["rows"][0]["group_id"] = "policy-0"
        with self.assertRaisesRegex(ValueError, "overlap"):
            report(c, artifact, policy)

    def test_namespace_cannot_hide_overlap(self):
        c = cache("test")
        c["group_namespace"] = "renamed"
        with self.assertRaisesRegex(ValueError, "namespace"):
            report(c, fit(cache()))

    def test_changed_model_or_recipe_is_rejected(self):
        c = cache("test")
        c["identity"]["precision"] = "int8"
        with self.assertRaisesRegex(ValueError, "different model"):
            report(c, fit(cache()))

    def test_policy_cannot_be_reused_with_changed_temperature(self):
        artifact = fit(cache())
        policy = select_policy(cache("policy"), artifact, .95)
        artifact["temperature"] = 1
        with self.assertRaisesRegex(ValueError, "different calibration"):
            report(cache("test"), artifact, policy)

    def test_logits_are_stable_and_nll_does_not_clip_wrong_predictions(self):
        row = cache(count=1)["rows"][0]
        row["logits"] = [1000, -1000]
        p, logp = probabilities(row)
        self.assertEqual(p, [1, 0])
        self.assertEqual(logp, [0, -2000])
        self.assertEqual(scored([row], 1)[0]["nll"], 2000)

    def test_probability_replay_matches_logits(self):
        row = cache(count=1)["rows"][0]
        p, _ = probabilities(row)
        q = dict(row, probabilities=p)
        q.pop("logits")
        for a, b in zip(probabilities(row, 2)[0], probabilities(q, 2)[0]):
            self.assertAlmostEqual(a, b)
        q["probabilities"] = [1, 0]
        with self.assertRaisesRegex(ValueError, "rounded zeros"):
            probabilities(q, 2)

    def test_per_type_fit_and_score_metrics(self):
        c = cache()
        for row in c["rows"]:
            row["type"] = "score"
        artifact = fit(c, per_type=True)
        self.assertEqual(set(artifact["temperature"]), {"score"})
        result = report(dict(cache("test"), rows=[dict(r, type="score") for r in cache("test")["rows"]]), artifact)
        self.assertIsNotNone(result["after"]["rps"])
        self.assertIn("2", result["groups"]["candidate_count"])

    def test_empty_fit_is_rejected(self):
        c = cache(count=1)
        c["rows"][0]["status"] = "error"
        with self.assertRaisesRegex(ValueError, "No valid"):
            fit(c)

    def test_invalid_cache_and_calibration_inputs(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "cache.json"
            cases = []
            c = cache(count=2); c["rows"][1]["id"] = c["rows"][0]["id"]; cases.append(c)
            c = cache(count=1); c["rows"][0]["target"] = [.8, .8]; cases.append(c)
            c = cache(count=1); c["rows"][0]["logits"] = [True, 1]; cases.append(c)
            c = cache(count=1); c["identity"]["artifacts_sha256"] = {}; cases.append(c)
            for c in cases:
                write_json(path, c)
                with self.assertRaises(ValueError): load_cache(path, "calibration")
            for value in [0, -1, True, "1", {"other": 1}]:
                artifact = dict(fit(cache()), temperature=value)
                write_json(path, artifact)
                with self.assertRaises(ValueError): load_calibration(path)

    def test_downloaded_aggregate_report_can_be_audited_and_replotted(self):
        result = report(cache("test"), fit(cache()))
        self.assertTrue(audit_report(result)["passed"])
        changed = copy.deepcopy(result)
        changed["after"]["nll"] += .1
        with self.assertRaisesRegex(ValueError, "arithmetic"):
            audit_report(changed)
        changed = copy.deepcopy(result)
        changed["after"]["risk_coverage"][-1]["errors"] += 1
        with self.assertRaises(ValueError):
            audit_report(changed)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "downloaded-report.json"
            write_json(path, result)
            main(["audit", str(path), "--output", str(Path(temp) / "replot")])
            self.assertTrue((Path(temp) / "replot/reliability.svg").is_file())
            self.assertTrue((Path(temp) / "replot/audit.json").is_file())

    def test_split_is_group_based_deterministic_and_not_score_based(self):
        c = cache("unassigned", 120)
        c["rows"][1]["group_id"] = c["rows"][0]["group_id"]
        parts = split_cache(c, "frozen-before-metrics")
        assignments = {r["id"]: role for role, part in parts.items() for r in part["rows"]}
        self.assertEqual(assignments["unassigned-0"], assignments["unassigned-1"])
        c["rows"].reverse()
        for row in c["rows"]: row["logits"] = [0, 100]
        changed = split_cache(c, "frozen-before-metrics")
        self.assertEqual(assignments, {r["id"]: role for role, part in changed.items() for r in part["rows"]})
        sets = [set(r["group_id"] for r in part["rows"]) for part in parts.values()]
        self.assertFalse(sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])

    def test_runtime_may_add_hashes_but_cannot_change_bound_files(self):
        expected = cache()["identity"]
        actual = copy.deepcopy(expected)
        actual["artifacts_sha256"]["extra-file"] = "c" * 64
        verify_runtime_identity(expected, actual)
        actual["artifacts_sha256"]["test-weights"] = "d" * 64
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            verify_runtime_identity(expected, actual)

    def test_pipeline_freezes_all_targets_and_writes_standalone_policies(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for role in ("calibration", "policy", "test"):
                write_json(root / (role + ".json"), cache(role))
            main(["pipeline", "--calibration-cache", str(root / "calibration.json"),
                  "--policy-cache", str(root / "policy.json"), "--test-cache", str(root / "test.json"),
                  "--max-error", ".01", "--max-error", ".95", "--output", str(root / "out")])
            result = json.loads((root / "out/report.json").read_text())
            self.assertEqual([p["validation_target"] for p in result["operating_points"]], [.01, .95])
            self.assertEqual(result["operating_points"][0]["test"]["accepted"], 0)
            self.assertEqual(result["operating_points"][1]["test"]["accepted"], 100)
            self.assertTrue((root / "out/policy-1.json").is_file())

    def test_complete_cli_pipeline_without_model_dependencies(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for role in ("calibration", "policy", "test"):
                write_json(root / (role + ".json"), cache(role))
            artifact, policy, out = root / "temperature.json", root / "rule.json", root / "report"
            main(["fit", str(root / "calibration.json"), "--output", str(artifact)])
            main(["select", str(root / "policy.json"), "--calibration", str(artifact),
                  "--max-error", ".95", "--output", str(policy)])
            main(["report", str(root / "test.json"), "--calibration", str(artifact),
                  "--policy", str(policy), "--output", str(out)])
            for name in ("report.json", "bins.csv", "risk_coverage.csv", "reliability.svg", "risk-coverage.svg"):
                self.assertTrue((out / name).is_file())
            result = json.loads((out / "report.json").read_text())
            self.assertEqual(result["test"]["questions"], 100)
            self.assertEqual(sum(b["n"] for b in result["after"]["bins"]), 100)


if __name__ == "__main__":
    unittest.main()
