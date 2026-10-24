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


def validate_response(response):
    if response["usage"]["output_tokens"] != 0 or response["usage"]["input_tokens"] < 1:
        raise ValueError("Invalid decision token usage")
    for answer in response["answers"].values():
        if answer["type"] == "noul":
            probabilities = [answer["noul"]]
        else:
            probabilities = list(answer["probabilities"].values())
            if not math.isclose(math.fsum(probabilities), 1.0, abs_tol=1e-6):
                raise ValueError("Decision probabilities do not sum to one")
            if (
                not math.isfinite(answer["confidence"])
                or not 0 <= answer["confidence"] <= 1
            ):
                raise ValueError("Invalid decision confidence")
            if (
                answer["type"] == "choice"
                and answer["choice"] not in answer["probabilities"]
            ):
                raise ValueError("Unknown selected candidate")
            if answer["type"] == "score":
                expected = sum(i * p for i, p in enumerate(probabilities))
                if not math.isfinite(answer["score"]) or not math.isclose(
                    answer["score"], expected, abs_tol=1e-6
                ):
                    raise ValueError("Score does not match its distribution")
        if not all(math.isfinite(p) and 0 <= p <= 1 for p in probabilities):
            raise ValueError("Invalid decision probabilities")


def decisions_request(request):
    """Express a parity case in the public protocol without changing its prompt."""
    state = copy.deepcopy(request["state"])
    if isinstance(state, dict):
        state = state["messages"]
        for message in state:
            if isinstance(message["content"], str):
                continue
            for part in message["content"]:
                if part["type"] == "text":
                    part["type"] = "input_text"
                else:
                    part["type"] = "input_image"
                    part["image_url"] = part["image_url"]["url"]
    questions = []
    for name, q in request["questions"].items():
        item = {
            "name": name,
            "type": "predicate" if q["type"] == "noul" else q["type"],
            "instructions": q["instructions"],
        }
        if q["type"] == "choice":
            item["choices"] = [
                {"value": key, "description": value}
                for key, value in q["criteria"].items()
            ]
        elif q["type"] == "score":
            item["levels"] = [
                {"label": str(i), "description": value}
                for i, value in enumerate(q["criteria"])
            ]
        questions.append(item)
    return {
        "model": request["model"],
        "input": state,
        "questions": questions,
        "temperature": request.get("temperature", 1.0),
    }


def normalize_decisions(request, response):
    """Check typed public answers and align them with the native parity schema."""
    response = dict(response)
    items = response["answers"]
    if not isinstance(items, list) or len(items) != len(request["questions"]):
        raise ValueError("Expected one ordered public answer per question")
    answers = {}
    for (name, q), item in zip(request["questions"].items(), items):
        kind = "predicate" if q["type"] == "noul" else q["type"]
        if item["name"] != name or item["type"] != kind:
            raise ValueError("Public answer order, name or type differs")
        if kind == "predicate":
            answer = {"type": "noul", "noul": item["probability"]}
        else:
            keys = (
                list(q["criteria"])
                if kind == "choice"
                else [str(i) for i in range(len(q["criteria"]))]
            )
            entries = item["probabilities"]
            labels = [
                entry["value"] if kind == "choice" else entry["label"]
                for entry in entries
            ]
            if labels != keys:
                raise ValueError("Public candidate values or labels differ")
            if kind == "score" and [entry["value"] for entry in entries] != list(
                range(len(keys))
            ):
                raise ValueError("Public score indices differ")
            answer = {
                "type": kind,
                "probabilities": {
                    key: entry["probability"] for key, entry in zip(keys, entries)
                },
                "confidence": item["confidence"],
                kind: item[kind],
            }
            if kind == "score":
                answer["legend"] = dict(zip(keys, q["criteria"]))
        answers[name] = answer
    response["answers"] = answers
    return response


def protocol_error(left, right):
    if left["usage"] != right["usage"] or left["model"] != right["model"]:
        raise ValueError("Protocols returned different usage or models")
    if left["answers"].keys() != right["answers"].keys():
        raise ValueError("Protocols returned different question keys")
    error = 0.0
    for key, old in left["answers"].items():
        new = right["answers"][key]
        if old["type"] != new["type"]:
            raise ValueError("Protocols returned different answer types")
        if old["type"] == "noul":
            error = max(error, abs(old["noul"] - new["noul"]))
        else:
            if old["probabilities"].keys() != new["probabilities"].keys():
                raise ValueError("Protocols returned different candidates")
            error = max(
                error,
                abs(old["confidence"] - new["confidence"]),
                *(
                    abs(p - new["probabilities"][k])
                    for k, p in old["probabilities"].items()
                ),
            )
            if old["type"] == "choice" and old["choice"] != new["choice"]:
                raise ValueError("Protocols selected different choices")
            if old["type"] == "score":
                error = max(error, abs(old["score"] - new["score"]))
    return error


def native(args, requests):
    import requests as http

    url = f"http://127.0.0.1:{args.port}"
    command = [
        sys.executable,
        "-m",
        "valen_vllm.launch_server",
        "--model-path",
        args.model_path,
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--max-model-len",
        "8192",
        "--gpu-memory-utilization",
        "0.65",
    ]
    (args.output_dir / "server-command.json").write_text(json.dumps(command, indent=2))
    log = (args.output_dir / "server.log").open("w")
    server = subprocess.Popen(
        command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
    )

    def score(request, protocol="systemone"):
        request = copy.deepcopy(request)
        if isinstance(request.get("state"), dict):
            import base64

            for message in request["state"]["messages"]:
                for part in (
                    message.get("content", [])
                    if isinstance(message.get("content"), list)
                    else []
                ):
                    if part.get("type") == "image_url":
                        path = Path(part["image_url"]["url"])
                        part["image_url"]["url"] = (
                            "data:image/png;base64,"
                            + base64.b64encode(path.read_bytes()).decode()
                        )
        start = time.perf_counter()
        native_request = {"model": "valen", **request}
        payload = (
            native_request
            if protocol == "systemone"
            else decisions_request(native_request)
        )
        response = http.post(url + "/v1/" + protocol, json=payload, timeout=180)
        response.raise_for_status()
        result = response.json()
        if protocol == "decisions":
            result = normalize_decisions(native_request, result)
        validate_response(result)
        return result, time.perf_counter() - start

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
            raise TimeoutError("vLLM did not become ready within 600 seconds")
        protocol_checks = []
        for index, request in enumerate(requests):
            for temperature in (0.5, 1.0, 2.0) if index == 0 else (1.0,):
                case = {**request, "temperature": temperature}
                old, _ = score(case, "systemone")
                public, _ = score(case, "decisions")
                error = protocol_error(old, public)
                protocol_checks.append(
                    {
                        "case": index,
                        "temperature": temperature,
                        "max_error": error,
                        "systemone": old,
                        "decisions": public,
                    }
                )
                if error > args.tolerance:
                    raise ValueError(f"Protocol parity error {error} exceeds tolerance")
        for protocol in ("systemone", "decisions"):
            invalid = {"model": "not-the-served-model", **requests[0]}
            payload = invalid if protocol == "systemone" else decisions_request(invalid)
            if (
                http.post(url + "/v1/" + protocol, json=payload, timeout=30).status_code
                != 404
            ):
                raise ValueError("Unknown model was not rejected")
        answers, timings = [], []
        for request in requests:
            score(request)
            runs = [score(request) for _ in range(args.repeats)]
            answers.append(runs[-1][0])
            timings.append(statistics.median(t for _, t in runs))
        concurrent_requests = [
            copy.deepcopy(requests[i % len(requests)]) for i in range(20)
        ]
        start = time.perf_counter()

        def concurrent_score(item):
            index, request = item
            return score(request, "decisions" if index % 2 else "systemone")

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            batch = list(pool.map(concurrent_score, enumerate(concurrent_requests)))
        elapsed = time.perf_counter() - start
        result = {
            "answers": answers,
            "median_seconds": timings,
            "protocol_checks": protocol_checks,
            "protocol_max_error": max(row["max_error"] for row in protocol_checks),
            "both_protocols_validated": True,
            "concurrent_protocols": [
                "decisions" if i % 2 else "systemone" for i in range(len(batch))
            ],
            "concurrent_answers": [a for a, _ in batch],
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
    if (
        not native.get("both_protocols_validated")
        or len(native.get("protocol_checks", [])) != 7
    ):
        raise ValueError("Expected all seven dual-protocol and temperature checks")
    protocol_max_error = max(
        protocol_error(row["systemone"], row["decisions"])
        for row in native["protocol_checks"]
    )
    if protocol_max_error > args.tolerance:
        raise ValueError("Dual-protocol parity failed")
    if set(native.get("concurrent_protocols", [])) != {"systemone", "decisions"}:
        raise ValueError("Expected mixed-protocol concurrent requests")
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
        if answer["answers"].keys() != expected["answers"].keys():
            raise ValueError("Concurrent response question keys differ")
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
    result = {
        "cases": rows,
        "both_protocols_validated": True,
        "protocol_max_error": protocol_max_error,
        "tokens_match": tokens_match,
        "choices_match": choices_match,
        "max_probability_error": max_error,
        "batch_max_probability_error": batch_error,
        "concurrent_requests_per_second": native["concurrent_requests_per_second"],
        "tolerance": args.tolerance,
        "passed": tokens_match
        and choices_match
        and max(max_error, batch_error) <= args.tolerance,
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
