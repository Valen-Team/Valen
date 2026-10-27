"""Offline reliability protocol v1. Standard library only; mirrored in Valen/JevalKit.

Caches contain raw logits or strictly positive raw probabilities, never confidence.
No model, network, or plotting dependency is needed to fit and replay a report.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

VERSION = 1


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def positive(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError("Temperature must be finite and positive")
    return value


def validate_identity(identity):
    for key in ("model", "revision", "backend", "precision", "protocol_sha256"):
        if not isinstance(identity.get(key), str) or not identity[key]:
            raise ValueError("Identity requires " + key)
    hashes = identity.get("artifacts_sha256")
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError("Identity requires weight/config artifact hashes")
    for value in [identity["protocol_sha256"], *hashes.values()]:
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError("Expected lowercase SHA-256 identities")


def load_cache(path, role):
    return validate_cache(read_json(path), role)


def validate_cache(cache, role):
    if cache.get("schema_version") != VERSION or cache.get("role") != role:
        raise ValueError("Cache role/schema mismatch; expected " + role)
    validate_identity(cache["identity"])
    if not isinstance(cache.get("dataset"), dict) or not cache["dataset"] or not isinstance(cache.get("group_namespace"), str) or not cache["group_namespace"] or not cache.get("rows"):
        raise ValueError("Cache needs dataset, stable group_namespace and nonempty rows")
    seen = set()
    for row in cache["rows"]:
        if not isinstance(row.get("id"), str) or not row["id"] or row["id"] in seen or not isinstance(row.get("group_id"), str) or not row["group_id"]:
            raise ValueError("Rows require unique IDs and stable source group IDs")
        seen.add(row["id"])
        if row.get("type") not in ("choice", "score", "noul"):
            raise ValueError("Unknown decision type")
        keys, target = row["candidates"], row["target"]
        if len(keys) < 2 or len(set(keys)) != len(keys) or not all(isinstance(k, str) for k in keys):
            raise ValueError("Require at least two distinct ordered candidates")
        if len(target) != len(keys) or any(isinstance(v, bool) or not isinstance(v, (int, float))
                or not math.isfinite(v) or v < 0 or v > 1 for v in target) or abs(sum(target) - 1) > 1e-8:
            raise ValueError("Invalid target distribution")
        if row.get("gold") is not None and row["gold"] not in keys:
            raise ValueError("Gold label is not a candidate")
        if row.get("status") not in ("ok", "error", "unsupported", "refusal"):
            raise ValueError("Explicit status required")
        if row["status"] == "ok":
            raw_logits(row)
    return cache


def raw_logits(row):
    if "logits" in row:
        values = row["logits"]
        if len(values) != len(row["candidates"]) or any(isinstance(v, bool) or not isinstance(v, (int, float))
                or not math.isfinite(v) for v in values):
            raise ValueError("Expected one finite raw logit per candidate")
        return values
    p = row.get("probabilities", [])
    if len(p) != len(row["candidates"]) or any(isinstance(v, bool) or not isinstance(v, (int, float))
            or not math.isfinite(v) or v <= 0 or v > 1 for v in p) or abs(sum(p) - 1) > 1e-8:
        raise ValueError("Replay requires raw logits or positive normalized probabilities; rounded zeros cannot be recovered")
    return [math.log(v) for v in p]


def probabilities(row, temperature=1):
    positive(temperature)
    z = raw_logits(row)
    top = max(z)
    z = [(v - top) / temperature for v in z]
    if not all(math.isfinite(v) for v in z):
        raise ValueError("Logit range/temperature overflow")
    normalizer = sum(math.exp(v) for v in z)
    logp = [v - math.log(normalizer) for v in z]
    return [math.exp(v) for v in logp], logp


def temperature_for(temperature, kind):
    if isinstance(temperature, dict) and kind not in temperature:
        raise ValueError("No fitted temperature for decision type: " + kind)
    return positive(temperature[kind] if isinstance(temperature, dict) else temperature)


def group_hashes(cache):
    return sorted({digest([cache["group_namespace"], row["group_id"]]) for row in cache["rows"]})


def provenance(cache):
    return {"dataset": cache["dataset"], "cache_sha256": digest(cache),
            "group_namespace": cache["group_namespace"], "group_hashes": group_hashes(cache),
            "questions": len(cache["rows"])}


def compatible(cache, artifact):
    if artifact.get("schema_version") != VERSION or cache["identity"] != artifact["identity"]:
        raise ValueError("Calibration/policy belongs to a different model or inference protocol")


def disjoint(cache, sources):
    groups = set(group_hashes(cache))
    for source in sources:
        if cache["group_namespace"] != source["group_namespace"]:
            raise ValueError("Use one stable source-entity group namespace across all splits")
        if groups.intersection(source["group_hashes"]):
            raise ValueError("Source groups overlap between calibration, policy or test splits")


def fit_temperature(rows):
    usable = [r for r in rows if r["status"] == "ok"]
    if not usable:
        raise ValueError("No valid calibration predictions")
    # NLL is convex in inverse temperature; bisection finds its bounded minimum.
    def derivative(beta):
        total = 0.0
        for row in usable:
            z = raw_logits(row)
            z = [v - max(z) for v in z]
            p, _ = probabilities(row, 1 / beta)
            total += sum((a - b) * v for a, b, v in zip(p, row["target"], z))
        return total / len(usable)
    lo, hi = .01, 100.0
    for _ in range(70):
        mid = (lo + hi) / 2
        if derivative(mid) > 0:
            hi = mid
        else:
            lo = mid
    return 1 / ((lo + hi) / 2)


def split_cache(cache, seed):
    """Retrospectively partition a raw cache by stable source group, never by score."""
    if cache.get("role") != "unassigned" or not seed:
        raise ValueError("Split requires an unassigned cache and a frozen nonempty seed")
    result = {}
    for role in ("calibration", "policy", "test"):
        result[role] = {**cache, "role": role, "rows": [],
                        "dataset": {**cache["dataset"], "partition": {"seed": seed, "role": role,
                                     "method": "SHA256(seed, group_id) modulo 100; 30/30/40"}}}
    for row in cache["rows"]:
        bucket = int(digest([seed, row["group_id"]])[:16], 16) % 100
        role = "calibration" if bucket < 30 else "policy" if bucket < 60 else "test"
        result[role]["rows"].append(row)
    if any(not part["rows"] for part in result.values()):
        raise ValueError("Too few source groups to populate all three splits")
    return result


def fit(cache, per_type=False):
    if cache.get("role") != "calibration":
        raise ValueError("Fit requires the calibration split")
    rows = cache["rows"]
    temperature = ({kind: fit_temperature([r for r in rows if r["type"] == kind])
                    for kind in sorted({r["type"] for r in rows})} if per_type else fit_temperature(rows))
    return {"schema_version": VERSION, "kind": "temperature", "identity": cache["identity"],
            "temperature": temperature, "objective": "mean soft-target NLL on valid calibration questions",
            "bounds": [.01, 100], "fit": provenance(cache)}


def scored(rows, temperature):
    values = []
    for row in rows:
        gold = row.get("gold")
        if gold is None and max(row["target"]) == 1:
            gold = row["candidates"][row["target"].index(1)]
        item = {"id": row["id"], "hard": gold is not None, "status": row["status"], "group_id": row["group_id"]}
        if row["status"] == "ok":
            p, logp = probabilities(row, temperature_for(temperature, row["type"]))
            top = max(range(len(p)), key=p.__getitem__)
            item.update(probabilities=p, predicted=row["candidates"][top], confidence=p[top], correct=row["candidates"][top] == gold if gold else None,
                        nll=-sum(t * v for t, v in zip(row["target"], logp)),
                        brier=sum((t - v) ** 2 for t, v in zip(row["target"], p)))
            if row["type"] == "score":
                item["mae"] = abs(sum(i * (a - b) for i, (a, b) in enumerate(zip(p, row["target"]))))
                item["rps"] = sum((sum(p[:i]) - sum(row["target"][:i])) ** 2 for i in range(1, len(p))) / (len(p) - 1)
        values.append(item)
    return values


def wilson(errors, n):
    if not n:
        return None
    z = 1.959963984540054
    p, q = errors / n, z * z
    center = (p + q / (2 * n)) / (1 + q / n)
    delta = z * math.sqrt(p * (1 - p) / n + q / (4 * n * n)) / (1 + q / n)
    return [max(0, center - delta), min(1, center + delta)]


def risk_curve(values):
    hard = [v for v in values if v["hard"]]
    valid = sorted((v for v in hard if v["status"] == "ok"), key=lambda v: -v["confidence"])
    points = [{"threshold": None, "accepted": 0, "errors": 0, "coverage": 0,
               "error_rate": None, "error_interval": None}]
    n, errors, i = 0, 0, 0
    while i < len(valid):
        threshold = valid[i]["confidence"]
        while i < len(valid) and valid[i]["confidence"] == threshold:
            n += 1
            errors += not valid[i]["correct"]
            i += 1
        points.append({"threshold": threshold, "accepted": n, "errors": errors,
                       "coverage": n / len(hard), "error_rate": errors / n, "error_interval": wilson(errors, n)})
    return points


def summary(values, bins=15):
    valid = [v for v in values if v["status"] == "ok"]
    hard = [v for v in values if v["hard"]]
    calibrated = [v for v in valid if v["hard"]]
    buckets = []
    for i in range(bins):
        bucket = [v for v in calibrated if min(int(v["confidence"] * bins), bins - 1) == i]
        n = len(bucket)
        buckets.append({"lower": i / bins, "upper": (i + 1) / bins, "n": n,
                        "mean_probability": sum(v["confidence"] for v in bucket) / n if n else None,
                        "accuracy": sum(v["correct"] for v in bucket) / n if n else None,
                        "accuracy_interval": wilson(sum(v["correct"] for v in bucket), n)})
    ece = sum(b["n"] * abs(b["mean_probability"] - b["accuracy"]) for b in buckets if b["n"])
    curve = risk_curve(values)
    out = {"questions": len(values), "valid": len(valid), "failed": len(values) - len(valid),
           "status_counts": {status: sum(v["status"] == status for v in values) for status in ("ok", "error", "unsupported", "refusal")},
           "hard_questions": len(hard), "ece_questions": len(calibrated),
           "valid_coverage": len(valid) / len(values) if values else 0,
           "accuracy": sum(bool(v.get("correct")) for v in hard) / len(hard) if hard else None,
           "ece": ece / len(calibrated) if calibrated else None, "bins": buckets,
           "probability_one_questions": sum(v["confidence"] == 1 for v in calibrated), "risk_coverage": curve,
           "aurc": sum((b["coverage"] - a["coverage"]) * b["error_rate"] for a, b in zip(curve, curve[1:])),
           "aurc_scope": "right-step integral over attainable coverage; failures are never accepted"}
    for metric in ("nll", "brier", "mae", "rps"):
        entries = [v[metric] for v in valid if metric in v]
        out[metric] = sum(entries) / len(entries) if entries else None
        out[metric + "_questions"] = len(entries)
        out[metric + "_sum"] = sum(entries)
    return out


def select_policy(cache, calibration, max_error, min_accepted=30):
    if cache.get("role") != "policy":
        raise ValueError("Select requires the policy-validation split")
    compatible(cache, calibration)
    disjoint(cache, [calibration["fit"]])
    if not 0 < max_error < 1 or min_accepted < 1:
        raise ValueError("Require 0 < max_error < 1 and min_accepted >= 1")
    values = scored(cache["rows"], calibration["temperature"])
    points = risk_curve(values)
    eligible = [p for p in points if p["accepted"] >= min_accepted and p["error_interval"][1] <= max_error]
    selected = max(eligible, key=lambda p: p["coverage"]) if eligible else points[0]
    return {"schema_version": VERSION, "kind": "top_probability_policy", "identity": cache["identity"],
            "calibration_sha256": digest(calibration), "calibration_fit": calibration["fit"],
            "selection": provenance(cache), "max_error": max_error, "min_accepted": min_accepted,
            "criterion": "Wilson 95% upper bound on policy validation; no test or shifted-domain guarantee",
            "threshold": selected["threshold"], "validation_operating_point": selected}


def evaluate_policy(values, policy):
    hard = [v for v in values if v["hard"]]
    accepted = [v for v in hard if v["status"] == "ok" and policy["threshold"] is not None
                and v["confidence"] >= policy["threshold"]]
    n, errors = len(accepted), sum(not v["correct"] for v in accepted)
    return {"threshold": policy["threshold"], "eligible": len(hard), "accepted": n, "errors": errors,
            "coverage": n / len(hard) if hard else 0, "manual_review": len(hard) - n,
            "error_rate": errors / n if n else None, "error_interval": wilson(errors, n)}


def report(cache, calibration, policy=None, bins=15):
    if cache.get("role") != "test":
        raise ValueError("Report requires the test split")
    if not 2 <= bins <= 100:
        raise ValueError("bins must be between 2 and 100")
    compatible(cache, calibration)
    sources = [calibration["fit"]]
    if policy:
        compatible(cache, policy)
        if policy["calibration_sha256"] != digest(calibration):
            raise ValueError("Policy was selected using a different calibration")
        sources.append(policy["selection"])
    disjoint(cache, sources)
    before, after = scored(cache["rows"], 1), scored(cache["rows"], calibration["temperature"])
    result = {"schema_version": VERSION, "identity": cache["identity"], "test": provenance(cache),
              "calibration": calibration, "policy": policy,
              "definitions": {"ece": f"top-label, {bins} equal-width bins, left-closed except final endpoint",
                              "brier": "multiclass sum, not divided by candidate count",
                              "confidence": "maximum candidate probability, not API concentration confidence",
                              "coverage": "accepted / all hard-label questions including failures",
                              "intervals": "pointwise Wilson 95%; assumes independent questions; not simultaneous or group-adjusted",
                              "ties": "first candidate; acceptance includes all equal probabilities",
                              "group_references": "same_as=overall refers to top-level before/after without duplicating identical curves"},
              "before": summary(before, bins), "after": summary(after, bins), "groups": {},
              "verification": {"paired_valid_questions": sum(v["status"] == "ok" for v in before),
                               "argmax_changes": sum(a.get("predicted") != b.get("predicted") for a, b in zip(before, after))}}
    for field in ("type", "modality", "domain", "language", "candidate_count"):
        grouped = {}
        for i, row in enumerate(cache["rows"]):
            key = str(len(row["candidates"])) if field == "candidate_count" else str(row.get(field, "unknown"))
            grouped.setdefault(key, []).append(i)
        result["groups"][field] = {key: ({"same_as": "overall"} if len(ids) == len(cache["rows"]) else
                                                {"before": summary([before[i] for i in ids], bins),
                                                 "after": summary([after[i] for i in ids], bins)})
                                          for key, ids in grouped.items()}
    if policy:
        result["operating_point"] = evaluate_policy(after, policy)
    return result


def svg_plot(path, title, series, diagonal=False):
    def point(x, y):
        return f"{60 + 480*x:.2f},{260 - 210*y:.2f}"
    elements = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 320" role="img">',
                f'<title>{title}</title>', '<rect width="600" height="320" fill="white"/>',
                '<path d="M60 50V260H540" fill="none" stroke="#555"/>']
    if diagonal:
        elements.append('<path d="M60 260L540 50" stroke="#999" stroke-dasharray="4"/>')
    for name, color, points in series:
        elements.append(f'<polyline points="{" ".join(point(x,y) for x,y in points)}" fill="none" stroke="{color}" stroke-width="2"/>')
        for x, y in points:
            elements.append(f'<circle cx="{60+480*x}" cy="{260-210*y}" r="3" fill="{color}"/>')
        elements.append(f'<text x="{70 if name == "before" else 200}" y="25" fill="{color}">{name}</text>')
    elements.extend([f'<text x="60" y="305" font-size="12">{title}</text>',
                     '<text x="50" y="280">0</text><text x="530" y="280">1</text><text x="35" y="55">1</text></svg>'])
    Path(path).write_text("\n".join(elements), encoding="utf-8")


def save_report(output, result):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "report.json", result)
    for key in ("bins", "risk_coverage"):
        rows = [dict(stage=stage, **row) for stage in ("before", "after") for row in result[stage][key]]
        with (output / (key + ".csv")).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    svg_plot(output / "reliability.svg", "Predicted probability (x) / observed accuracy (y)",
             [(s, c, [(b["mean_probability"], b["accuracy"]) for b in result[s]["bins"] if b["n"]])
              for s, c in (("before", "#777"), ("after", "#3964fe"))], True)
    svg_plot(output / "risk-coverage.svg", "Automated coverage (x) / accepted error rate (y)",
             [(s, c, [(p["coverage"], p["error_rate"]) for p in result[s]["risk_coverage"] if p["accepted"]])
              for s, c in (("before", "#777"), ("after", "#3964fe"))])


def audit_report(result):
    """Check aggregate arithmetic without weights or private per-question caches.

    This checks the report's internal consistency, not that an input model or
    dataset is authentic. Full experiment replay still requires retained inputs.
    """
    checked = []
    def close(a, b, name):
        if a is None or b is None:
            if a is not b:
                raise ValueError("Report arithmetic mismatch: " + name)
        elif not math.isfinite(a) or not math.isfinite(b) or not math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-10):
            raise ValueError("Report arithmetic mismatch: " + name)
    for stage in ("before", "after"):
        metrics = result[stage]
        n = sum(b["n"] for b in metrics["bins"])
        if n != metrics["ece_questions"] or n > metrics["hard_questions"]:
            raise ValueError("Report bin denominator mismatch")
        total_gap = sum(b["n"] * abs(b["mean_probability"] - b["accuracy"]) for b in metrics["bins"] if b["n"])
        close(metrics["ece"], total_gap / n if n else None, stage + " ECE")
        correct = sum(b["n"] * b["accuracy"] for b in metrics["bins"] if b["n"])
        hard = metrics["hard_questions"]
        close(metrics["accuracy"], correct / hard if hard else None, stage + " accuracy")
        for key in ("nll", "brier", "mae", "rps"):
            count = metrics[key + "_questions"]
            close(metrics[key], metrics[key + "_sum"] / count if count else None, stage + " " + key)
        curve = metrics["risk_coverage"]
        if not curve or curve[0]["accepted"] != 0 or curve[0]["error_rate"] is not None:
            raise ValueError("Report requires an explicit accept-nothing point")
        previous = curve[0]
        for point in curve[1:]:
            accepted, errors = point["accepted"], point["errors"]
            if not previous["accepted"] < accepted <= n or not previous["errors"] <= errors <= accepted:
                raise ValueError("Invalid cumulative risk counts")
            if errors - previous["errors"] > accepted - previous["accepted"]:
                raise ValueError("Too many errors in risk increment")
            if previous["threshold"] is not None and not point["threshold"] < previous["threshold"]:
                raise ValueError("Risk thresholds must decrease without splitting ties")
            close(point["coverage"], accepted / hard, stage + " coverage")
            close(point["error_rate"], errors / accepted, stage + " risk")
            previous = point
        if curve[-1]["accepted"] != n:
            raise ValueError("Risk curve omitted valid hard-label questions")
        close(metrics["aurc"], sum((b["coverage"] - a["coverage"]) * b["error_rate"]
                                   for a, b in zip(curve, curve[1:])), stage + " AURC")
        checked.append(stage + ": ECE, accuracy, proper-score means, risk counts, coverage and AURC")
    return {"passed": True, "checks": checked,
            "scope": "aggregate arithmetic only; original prediction/data hashes and full replay establish input provenance"}


def pipeline(calibration_cache, policy_cache, test_cache, output, budgets=None, per_type=False, min_accepted=30):
    """Freeze several validation rules, then evaluate each on the same independent test."""
    artifact = fit(calibration_cache, per_type)
    budgets = sorted(set(budgets or [.01, .02, .05, .10]))
    policies = [select_policy(policy_cache, artifact, budget, min_accepted) for budget in budgets]
    result = report(test_cache, artifact, policies[0])
    values = scored(test_cache["rows"], artifact["temperature"])
    result["operating_points"] = [{"validation_target": p["max_error"], "policy": p,
                                   "test": evaluate_policy(values, p)} for p in policies]
    save_report(output, result)
    write_json(Path(output) / "calibration.json", artifact)
    for i, policy in enumerate(policies):
        write_json(Path(output) / f"policy-{i}.json", policy)
    return result


def verify_runtime_identity(expected, actual):
    """Allow extra runtime fingerprints, but require every bound artifact to match."""
    validate_identity(expected)
    validate_identity(actual)
    for key in ("model", "revision", "backend", "precision", "protocol_sha256"):
        if expected[key] != actual[key]:
            raise ValueError("Calibration model/backend/protocol identity mismatch: " + key)
    for name, fingerprint in expected["artifacts_sha256"].items():
        if actual["artifacts_sha256"].get(name) != fingerprint:
            raise ValueError("Calibration artifact fingerprint mismatch: " + name)


def load_calibration(path, identity=None):
    artifact = read_json(path)
    if artifact.get("schema_version") != VERSION or artifact.get("kind") != "temperature":
        raise ValueError("Expected a versioned temperature calibration artifact")
    validate_identity(artifact["identity"])
    temperatures = artifact["temperature"]
    if isinstance(temperatures, dict):
        if not temperatures or set(temperatures) - {"choice", "score", "noul"}:
            raise ValueError("Invalid per-type temperatures")
        for t in temperatures.values():
            positive(t)
    else:
        positive(temperatures)
    if identity is not None:
        verify_runtime_identity(artifact["identity"], identity)
    return artifact


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    audit = sub.add_parser("audit", help="Verify report arithmetic and optionally replot aggregate evidence offline")
    audit.add_argument("report")
    audit.add_argument("--output")
    partition = sub.add_parser("split", help="Retrospective 30/30/40 source-group split of a raw cache")
    partition.add_argument("cache")
    partition.add_argument("--seed", required=True, help="Freeze before inspecting metrics")
    partition.add_argument("--output", required=True)
    f = sub.add_parser("fit")
    f.add_argument("cache")
    f.add_argument("--per-type", action="store_true")
    f.add_argument("--output", required=True)
    s = sub.add_parser("select")
    s.add_argument("cache")
    s.add_argument("--calibration", required=True)
    s.add_argument("--max-error", type=float, required=True)
    s.add_argument("--min-accepted", type=int, default=30)
    s.add_argument("--output", required=True)
    r = sub.add_parser("report")
    r.add_argument("cache")
    r.add_argument("--calibration", required=True)
    r.add_argument("--policy")
    r.add_argument("--bins", type=int, default=15)
    r.add_argument("--output", required=True)
    batch = sub.add_parser("pipeline", help="Fit, freeze validation policies, and evaluate independent test")
    batch.add_argument("--calibration-cache", required=True)
    batch.add_argument("--policy-cache", required=True)
    batch.add_argument("--test-cache", required=True)
    batch.add_argument("--output", required=True)
    batch.add_argument("--max-error", type=float, action="append")
    batch.add_argument("--min-accepted", type=int, default=30)
    batch.add_argument("--per-type", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "audit":
        result = read_json(args.report)
        verification = audit_report(result)
        if args.output:
            save_report(args.output, result)
            write_json(Path(args.output) / "audit.json", verification)
        else:
            print(json.dumps(verification))
    elif args.command == "split":
        for role, part in split_cache(load_cache(args.cache, "unassigned"), args.seed).items():
            write_json(Path(args.output) / (role + ".json"), part)
    elif args.command == "pipeline":
        pipeline(load_cache(args.calibration_cache, "calibration"), load_cache(args.policy_cache, "policy"),
                 load_cache(args.test_cache, "test"), args.output, args.max_error, args.per_type, args.min_accepted)
    elif args.command == "fit":
        write_json(args.output, fit(load_cache(args.cache, "calibration"), args.per_type))
    elif args.command == "select":
        write_json(args.output, select_policy(load_cache(args.cache, "policy"), load_calibration(args.calibration),
                                             args.max_error, args.min_accepted))
    else:
        save_report(args.output, report(load_cache(args.cache, "test"), load_calibration(args.calibration),
                                        read_json(args.policy) if args.policy else None, args.bins))


if __name__ == "__main__":
    main()
