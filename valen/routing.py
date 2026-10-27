"""Apply saved calibration and routing policies to native decision logits."""
from valen.evaluation.calibration import digest, probabilities


def route_choice(keys, logits, calibration, policy):
    if policy.get("kind") != "top_probability_policy" or policy["identity"] != calibration["identity"]:
        raise ValueError("Policy/model identity mismatch")
    if policy["calibration_sha256"] != digest(calibration):
        raise ValueError("Policy was selected using a different temperature artifact")
    if policy["threshold"] is None:
        return {"route": "human_review", "reason": "no_rule_met_validation_target"}
    temperature = calibration["temperature"]
    if isinstance(temperature, dict):
        temperature = temperature["choice"]
    p, _ = probabilities({"candidates": keys, "logits": logits}, temperature)
    top = max(range(len(p)), key=p.__getitem__)
    return {"route": "automatic" if p[top] >= policy["threshold"] else "human_review",
            "label": keys[top], "probability": p[top], "validation_target": policy["max_error"],
            "threshold": policy["threshold"]}
