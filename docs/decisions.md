# Inference protocols

Valen supports two public inference protocols: **SystemOne** (Jev-style) and
**Decisions** (OpenAI-style). They express the same supported decision tasks with
different request and answer structures.

| | SystemOne | Decisions |
|---|---|---|
| Local HTTP route | `/v1/systemone` | `/v1/decisions` |
| Input | `state` | `input` |
| Questions | Mapping keyed by question ID | Ordered list with `name` |
| Predicate | `noul` | `predicate` |
| Choice candidates | `criteria` mapping | `choices` list |
| Score levels | Ordered `criteria` list | `levels` list with labels |
| Answers | Mapping keyed by question ID | Ordered list with `name` and `type` |

HF `model.predict(request)` and SDK `client.predict(request)` select the protocol from the
request fields and return the matching answer structure. SGLang, vLLM and the Python ONNX server select it by HTTP route.
The hosted API documents both protocols at the [API reference](https://valen-team.github.io/docs/valen/api/). Use the served model name configured by your deployment.

The SGLang and vLLM plugins expose both routes through the native Valen decision
head. See the [SGLang](../serving/sglang/README.md) and
[vLLM](../serving/vllm/README.md) deployment instructions. For published HF models,
see [Transformers inference](https://valen-team.github.io/docs/valen/quickstart/). For Python / JavaScript ONNX inference and the local ONNX service, see [ONNX & SDKs](onnx.md).

## SystemOne

```json
{
  "model": "valen",
  "state": "A cat is on the sofa.",
  "questions": {
    "on_sofa": {"type": "noul", "instructions": "The cat is on the sofa."},
    "animal": {"type": "choice", "instructions": "Which animal is present?",
               "criteria": {"cat": "A cat", "dog": "A dog"}},
    "clarity": {"type": "score", "instructions": "How clearly is the animal identified?",
                "criteria": ["Unclear", "Clear"]}
  }
}
```

Read `response["answers"]["on_sofa"]["noul"]` for the predicate probability.
Choice probabilities are a mapping keyed by candidate name. Score probabilities
are keyed by zero-based level strings, with an expected `score` and `confidence`.
HF uses the same payload without requiring `model`.

## Decisions

```json
{
  "model": "valen",
  "input": "A cat is on the sofa.",
  "temperature": 1.0,
  "questions": [
    {"name": "on_sofa", "type": "predicate", "instructions": "The cat is on the sofa."},
    {"name": "animal", "type": "choice", "instructions": "Which animal is present?",
     "choices": [{"value": "cat", "description": "A cat"}, {"value": "dog", "description": "A dog"}]},
    {"name": "clarity", "type": "score", "instructions": "How clearly is the animal identified?",
     "levels": [{"label": "unclear", "description": "Unclear"}, {"label": "clear", "description": "Clear"}]}
  ]
}
```

HTTP requests require the configured served model name. Local HF inference selects
its already loaded model. Question names are unique; an omitted name defaults to
its zero-based question index as a string. Choice values can be strings or booleans.
Score levels are ordered, with values `0` through `len(levels)-1`. Descriptions and
instructions determine the model prompt; question names and score labels are response metadata. Choice values are
candidate keys in the trained prompt. Temperature must be finite and positive and scales the
candidate logits before softmax.

The response contains an ordered `answers` list. Each answer has `name` and `type`:

- `predicate`: `probability` that the statement is true.
- `choice`: `choice`, `confidence`, and `probabilities: [{value, probability}]`.
- `score`: expected `score`, `confidence`, and `probabilities: [{label, value, probability}]`.

The response retains the backend's `model`, `usage` and available metadata. Hosted
fields such as `model_id`, `revision`, timing and billing counters depend on the
hosting service; local runtimes do not synthesize them. Output token count is zero
because the model scores candidates without generating text.

For images, replace `input` with messages:

```json
[{"role":"user","content":[
  {"type":"input_text","text":"Inspect this image."},
  {"type":"input_image","image_url":"data:image/png;base64,..."}
]}]
```

PNG/JPEG/WebP base64 data URLs work across the three runtimes. vLLM retains its
existing inline-image limits (16 images, 24 MiB encoded and 32 megapixels each).
The local HF compiler also accepts image paths relative to `media_root`; it does
not fetch remote URLs. For local HF video inference, use an `input_video` part:

```json
[{"role":"user","content":[
  {"type":"input_video","video_url":"clip.mp4"}
]}]
```

HF resolves the video path against `media_root` and uses the checkpoint's frame
sampling settings. HTTP serving accepts extracted frames as image inputs.

## Protocol choice and model capabilities

Both protocols support predicate/noul, choice, and score. Decisions responses
preserve score labels and boolean choice values. SystemOne uses string candidate
keys and ordinal score levels.

Training JSONL and evaluation code use the `state` / `criteria` schema. This is
separate from which public protocol an inference client chooses. Both HF and Python / JavaScript SDK protocols
support `shared_state` and `question` execution. A request-body `temperature`
overrides the keyword argument; either defaults to `1.0`.

## Validation

Run the CPU contract checks from the repository root:

```bash
PYTHONPATH=.:serving/vllm/src python -m unittest discover -s tests/common -p test_public_decisions.py -v
```

These checks cover conversion, validation, temperature, image cleanup, HTTP
adapters and the exported predict method using mocked model outputs. They do not
replace GPU engine validation.
