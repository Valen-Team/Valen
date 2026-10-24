"""Compare native Valen serving with the checkpoint's Transformers implementation.

Run on one CUDA GPU with the checkpoint already downloaded:
  python validate.py --model-path /models/Valen-2B --output-dir /tmp/valen-check

The reference and server run in separate processes so neither retains the other's
GPU allocations. Results include request token counts, probability differences,
single-request latency, and throughput with concurrent HTTP requests.
"""

import argparse
import concurrent.futures
import copy
import json
import math
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path


def cases(output):
    from PIL import Image, ImageDraw

    image = output / "red-square.png"
    canvas = Image.new("RGB", (512, 384), "white")
    ImageDraw.Draw(canvas).rectangle((128, 64, 384, 320), fill="red")
    canvas.save(image)
    questions = {
        "team": {
            "type": "choice",
            "instructions": "Which team should handle the request?",
            "criteria": {
                "billing": "Payments and refunds",
                "support": "Technical support",
                "sales": "New purchases",
            },
        },
        "urgent": {
            "type": "noul",
            "instructions": "Does the user need immediate emergency assistance?",
        },
        "severity": {
            "type": "score",
            "instructions": "How severe is the problem?",
            "criteria": ["Minor", "Moderate", "Severe"],
        },
    }
    text = {
        "state": "I was charged twice for my order. Please refund the duplicate payment.",
        "questions": questions,
    }
    visual = {
        "state": {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": str(image.resolve())},
                        },
                        {
                            "type": "text",
                            "text": "Describe the main shape in this image.",
                        },
                    ],
                }
            ]
        },
        "questions": {
            "color": {
                "type": "choice",
                "instructions": "What is the color of the main shape?",
                "criteria": {"red": "Red", "blue": "Blue", "green": "Green"},
            },
            "square": {
                "type": "noul",
                "instructions": "Is there a square in the image?",
            },
            "redness": {
                "type": "score",
                "instructions": "How red is the main shape?",
                "criteria": ["Not red", "Partly red", "Fully red"],
            },
        },
    }
    dialogue = {
        "state": {
            "messages": [
                {"role": "user", "content": "My screen is black."},
                {"role": "assistant", "content": "Is the power cable connected?"},
                {"role": "user", "content": "Yes, but it still does not work."},
            ]
        },
        "questions": questions,
    }
    many_levels = {
        "state": "There are three red apples and two green apples.",
        "questions": {
            "count": {
                "type": "score",
                "instructions": "How many apples are there?",
                "criteria": [f"{n} apples" for n in range(11)],
            }
        },
    }
    long_text = copy.deepcopy(text)
    long_text["state"] = (
        "Previous order notes: the first order was delivered successfully. " * 120
    ) + text["state"]
    return [text, dialogue, many_levels, long_text, visual]


def decisions_request(request):
    """Express a SystemOne parity fixture using the public Decisions schema."""
    state = copy.deepcopy(request["state"])
    if isinstance(state, dict):
        state = state["messages"]
        for message in state:
            if not isinstance(message["content"], list):
                continue
            for part in message["content"]:
                if part["type"] == "text":
                    part["type"] = "input_text"
                else:
                    part["type"] = "input_image"
                    part["image_url"] = part["image_url"]["url"]
    questions = []
    for name, q in request["questions"].items():
        question = {
            "name": name,
            "type": "predicate" if q["type"] == "noul" else q["type"],
            "instructions": q["instructions"],
        }
        if q["type"] == "choice":
            question["choices"] = [
                {"value": value, "description": text}
                for value, text in q["criteria"].items()
            ]
        elif q["type"] == "score":
            question["levels"] = [
                {"label": str(i), "description": text}
                for i, text in enumerate(q["criteria"])
            ]
        questions.append(question)
    return {
        "input": state,
        "questions": questions,
        "temperature": request.get("temperature", 1.0),
    }


def systemone_response(response):
    """Normalize a Decisions result for the existing HF comparison code."""
    result = dict(response)
    answers = {}
    for answer in response["answers"]:
        kind = answer["type"]
        entry = {"type": "noul" if kind == "predicate" else kind}
        if kind == "predicate":
            entry["noul"] = answer["probability"]
        else:
            entry["probabilities"] = {
                str(p["value"]): p["probability"] for p in answer["probabilities"]
            }
            entry[kind] = answer[kind]
            entry["confidence"] = answer["confidence"]
            if kind == "score":
                entry["legend"] = {
                    str(p["value"]): p["label"] for p in answer["probabilities"]
                }
        answers[answer["name"]] = entry
    result["answers"] = answers
    return result


def protocol_difference(expected, actual):
    """Check types, identities, distributions and non-probability readouts."""
    if list(expected["answers"]) != list(actual["answers"]):
        raise ValueError("Protocol question order/identity differs")
    if expected["usage"] != actual["usage"]:
        raise ValueError("Protocol token counts differ")
    errors = []
    for key, ref in expected["answers"].items():
        got = actual["answers"][key]
        if ref["type"] != got["type"]:
            raise ValueError("Protocol question types differ")
        if ref["type"] == "noul":
            errors.append(abs(ref["noul"] - got["noul"]))
        else:
            if list(ref["probabilities"]) != list(got["probabilities"]):
                raise ValueError("Protocol candidate identities/order differ")
            errors.extend(
                abs(p - got["probabilities"][k])
                for k, p in ref["probabilities"].items()
            )
            errors.append(abs(ref["confidence"] - got["confidence"]))
            if ref["type"] == "choice" and ref["choice"] != got["choice"]:
                raise ValueError("Protocol selected choices differ")
            if ref["type"] == "score":
                errors.append(abs(ref["score"] - got["score"]))
    if not all(math.isfinite(e) for e in errors):
        raise ValueError("Non-finite protocol result")
    return max(errors, default=0.0)


def reference(args, requests):
    import torch
    from transformers import AutoModel, AutoProcessor

    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    model = (
        AutoModel.from_pretrained(args.model_path, trust_remote_code=True, dtype="auto")
        .eval()
        .cuda()
    )
    answers, timings = [], []
    for request in requests:
        model.predict(request, processor=processor)
        latencies = []
        for _ in range(args.repeats):
            torch.cuda.synchronize()
            start = time.perf_counter()
            answer = model.predict(request, processor=processor)
            torch.cuda.synchronize()
            latencies.append(time.perf_counter() - start)
        answers.append(answer)
        timings.append(statistics.median(latencies))
    result = {
        "answers": answers,
        "median_seconds": timings,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "weight_dtype": str(next(model.backbone.parameters()).dtype),
    }
    (args.output_dir / "reference.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False)
    )


def native(args, requests):
    import requests as http

    url = f"http://127.0.0.1:{args.port}"
    command = [
        sys.executable,
        "-m",
        "valen_sglang.launch_server",
        "--model-path",
        args.model_path,
        "--served-model-name",
        "valen",
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--context-length",
        "8192",
        "--max-prefill-tokens",
        "16384",
        "--mem-fraction-static",
        "0.65",
        "--attention-backend",
        "fa3",
    ]
    (args.output_dir / "server-command.json").write_text(json.dumps(command, indent=2))
    log = (args.output_dir / "server.log").open("w")
    server = subprocess.Popen(
        command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
    )

    def score(request, protocol="systemone"):
        start = time.perf_counter()
        body = request if protocol == "systemone" else decisions_request(request)
        response = http.post(
            url + "/v1/" + protocol, json={"model": "valen", **body}, timeout=180
        )
        response.raise_for_status()
        result = response.json()
        return (
            result if protocol == "systemone" else systemone_response(result)
        ), time.perf_counter() - start

    try:
        deadline = time.monotonic() + 600
        while time.monotonic() < deadline:
            if server.poll() is not None:
                raise RuntimeError(
                    f"Server exited with {server.returncode}; see server.log"
                )
            try:
                response = http.get(url + "/health", timeout=2)
                if response.status_code == 200:
                    break
            except http.RequestException:
                pass
            time.sleep(2)
        else:
            raise TimeoutError("SGLang did not become ready within 600 seconds")
        answers, timings = [], []
        for request in requests:
            score(request)
            runs = [score(request) for _ in range(args.repeats)]
            answers.append(runs[-1][0])
            timings.append(statistics.median(t for _, t in runs))
        decisions_answers = [score(request, "decisions")[0] for request in requests]
        protocol_errors = [
            protocol_difference(a, b) for a, b in zip(answers, decisions_answers)
        ]
        # Temperature must affect logits in both protocols, not just be accepted.
        temperatures = []
        for temperature in (0.5, 2.0):
            request = dict(requests[0], temperature=temperature)
            left, right = score(request)[0], score(request, "decisions")[0]
            temperatures.append(
                {
                    "temperature": temperature,
                    "difference": protocol_difference(left, right),
                    "answers": left["answers"],
                }
            )
        if temperatures[0]["answers"] == temperatures[1]["answers"]:
            raise ValueError("Temperature did not change the model distribution")
        # Typed values and score labels must survive the Decisions boundary.
        typed = decisions_request(requests[0])
        typed["questions"][0]["choices"] = [
            {"value": True, "description": "Payments and refunds"},
            {"value": False, "description": "Technical support"},
        ]
        typed["questions"][2]["levels"] = [
            {"label": label, "description": text}
            for label, text in zip(
                ["low", "medium", "high"],
                requests[0]["questions"]["severity"]["criteria"],
            )
        ]
        response = http.post(
            url + "/v1/decisions", json={"model": "valen", **typed}, timeout=180
        )
        response.raise_for_status()
        typed_answer = response.json()
        if not isinstance(typed_answer["answers"][0]["choice"], bool):
            raise ValueError("Boolean choice value was lost")
        if [p["label"] for p in typed_answer["answers"][2]["probabilities"]] != [
            "low",
            "medium",
            "high",
        ]:
            raise ValueError("Score labels were lost")
        invalid = []
        for route, body in [
            ("decisions", {"model": "missing", **decisions_request(requests[0])}),
            ("systemone", {"model": "missing", **requests[0]}),
            (
                "decisions",
                {"model": "valen", **decisions_request(requests[0]), "temperature": 0},
            ),
        ]:
            response = http.post(url + "/v1/" + route, json=body, timeout=30)
            if response.status_code < 400 or response.status_code >= 500:
                raise ValueError(
                    f"Invalid {route} request returned {response.status_code}: {response.text[:400]}"
                )
            invalid.append({"route": route, "status": response.status_code})
        concurrent_requests = [
            (
                copy.deepcopy(requests[i % len(requests)]),
                "systemone" if i % 2 == 0 else "decisions",
            )
            for i in range(20)
        ]
        start = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            batch = list(pool.map(lambda pair: score(*pair), concurrent_requests))
        elapsed = time.perf_counter() - start
        result = {
            "answers": answers,
            "median_seconds": timings,
            "concurrent_answers": [a for a, _ in batch],
            "decisions_answers": decisions_answers,
            "protocol_errors": protocol_errors,
            "temperature_cases": temperatures,
            "typed_answer": typed_answer,
            "invalid_requests": invalid,
            "concurrent_seconds": elapsed,
            "concurrent_requests_per_second": len(batch) / elapsed,
        }
        (args.output_dir / "native.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False)
        )
    finally:
        import signal

        os.killpg(server.pid, signal.SIGTERM) if server.poll() is None else None
        try:
            server.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(server.pid, signal.SIGKILL)
            server.wait()
        log.close()


def compare(args):
    reference = json.loads((args.output_dir / "reference.json").read_text())
    native = json.loads((args.output_dir / "native.json").read_text())
    if len(reference["answers"]) != 5 or len(native["answers"]) != 5:
        raise ValueError("Expected all five parity cases")
    if len(native["concurrent_answers"]) != 20:
        raise ValueError("Expected all twenty concurrent responses")
    rows = []
    max_error = 0.0
    tokens_match = True
    choices_match = True
    for i, (expected, actual) in enumerate(
        zip(reference["answers"], native["answers"])
    ):
        tokens_match &= (
            expected["usage"]["input_tokens"] == actual["usage"]["input_tokens"]
        )
        if expected["answers"].keys() != actual["answers"].keys():
            raise ValueError("Native response question keys differ from reference")
        error = 0.0
        for key, answer in expected["answers"].items():
            got = actual["answers"][key]
            if answer["type"] != got["type"]:
                raise ValueError("Native response answer type differs from reference")
            values = (
                [got["noul"]]
                if got["type"] == "noul"
                else list(got["probabilities"].values())
            )
            if not all(math.isfinite(value) for value in values):
                raise ValueError("Non-finite native probability")
            if answer["type"] == "noul":
                error = max(error, abs(answer["noul"] - got["noul"]))
            else:
                error = max(
                    error,
                    *(
                        abs(p - got["probabilities"][k])
                        for k, p in answer["probabilities"].items()
                    ),
                )
            if answer["type"] == "choice":
                choices_match &= answer["choice"] == got["choice"]
        max_error = max(error, max_error)
        rows.append(
            {
                "case": i,
                "prompt_tokens": expected["usage"]["input_tokens"],
                "max_probability_error": error,
                "reference_seconds": reference["median_seconds"][i],
                "native_seconds": native["median_seconds"][i],
                "speedup": reference["median_seconds"][i] / native["median_seconds"][i],
            }
        )
    batch_error = 0.0
    for i, answer in enumerate(native["concurrent_answers"]):
        expected = native["answers"][i % len(rows)]
        for key, got in answer["answers"].items():
            ref = expected["answers"][key]
            values = (
                [abs(got["noul"] - ref["noul"])]
                if got["type"] == "noul"
                else [
                    abs(p - ref["probabilities"][k])
                    for k, p in got["probabilities"].items()
                ]
            )
            batch_error = max(batch_error, *values)
    protocol_error = max(
        native["protocol_errors"]
        + [c["difference"] for c in native["temperature_cases"]]
    )
    result = {
        "cases": rows,
        "protocol_max_error": protocol_error,
        "both_protocols_validated": True,
        "tokens_match": tokens_match,
        "choices_match": choices_match,
        "max_probability_error": max_error,
        "batch_max_probability_error": batch_error,
        "concurrent_requests_per_second": native["concurrent_requests_per_second"],
        "tolerance": args.tolerance,
        "passed": tokens_match
        and choices_match
        and max(max_error, batch_error) <= args.tolerance
        and protocol_error <= args.protocol_tolerance,
    }
    (args.output_dir / "comparison.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)
    if not result["passed"]:
        raise SystemExit("Valen parity check failed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--mode", choices=["all", "reference", "native", "compare"], default="all"
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--port", type=int, default=31000)
    parser.add_argument("--tolerance", type=float, default=0.02)
    parser.add_argument("--protocol-tolerance", type=float, default=1e-6)
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    requests = cases(args.output_dir)
    if args.mode == "all":
        for mode in ("reference", "native", "compare"):
            subprocess.run(
                [
                    sys.executable,
                    __file__,
                    "--model-path",
                    args.model_path,
                    "--output-dir",
                    str(args.output_dir),
                    "--mode",
                    mode,
                    "--repeats",
                    str(args.repeats),
                    "--port",
                    str(args.port),
                    "--tolerance",
                    str(args.tolerance),
                    "--protocol-tolerance",
                    str(args.protocol_tolerance),
                ],
                check=True,
            )
    elif args.mode == "reference":
        reference(args, requests)
    elif args.mode == "native":
        native(args, requests)
    else:
        compare(args)


if __name__ == "__main__":
    main()
