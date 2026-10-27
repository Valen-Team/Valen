<p align="center">
  <img src="assets/branding/valen-mark-white.png" alt="Valen" width="112">
</p>
<h1 align="center">Valen</h1>
<p align="center"><strong>Fast Decisions. Continuous Evolution.</strong></p>
<p align="center">A multimodal System One decision model. Text, images and video in; structured decision probabilities out.</p>

<p align="center">
  <a href="https://huggingface.co/Valen-Team/Valen-0.8B"><img src="https://img.shields.io/badge/Valen-0.8B-3964FE?style=flat&amp;labelColor=1E232C" alt="Valen 0.8B"></a>
  <a href="https://huggingface.co/Valen-Team/Valen-2B"><img src="https://img.shields.io/badge/Valen-2B-3964FE?style=flat&amp;labelColor=1E232C" alt="Valen 2B"></a>
  <a href="https://huggingface.co/Valen-Team/Valen-4B"><img src="https://img.shields.io/badge/Valen-4B-3964FE?style=flat&amp;labelColor=1E232C" alt="Valen 4B"></a>
  <a href="https://huggingface.co/datasets/Valen-Team/VisualDecisionBench"><img src="https://img.shields.io/badge/Dataset-VisualDecisionBench-3964FE?style=flat&amp;labelColor=1E232C" alt="VisualDecisionBench"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-555555?style=flat&amp;labelColor=1E232C" alt="Apache 2.0"></a>
  <a href="pyproject.toml"><img src="https://img.shields.io/badge/Python-3.10%2B-555555?style=flat&amp;labelColor=1E232C" alt="Python 3.10+"></a>
</p>

<p align="center">
  <strong>English</strong> · <a href="README_zh.md">简体中文</a><br>
  <a href="https://valen-team.github.io/">Try Valen</a> · <a href="https://valen-team.github.io/docs/">Documentation</a> · <a href="https://valen-team.github.io/docs/valen/api/">API</a> · <a href="https://huggingface.co/Valen-Team">Hugging Face</a>
</p>

<p align="center">
  <a href="https://github.com/Valen-Team/valen-sdk-python">Python SDK</a> · <a href="https://github.com/Valen-Team/valen-sdk-js">JavaScript SDK</a> · <a href="https://github.com/Valen-Team/valen-sdk-cpp">C++ text integration</a>
</p>

<p align="center">
  <a href="#introduction">Introduction</a> · <a href="#version-selection">Choose a version</a> · <a href="#model-downloads">Models &amp; datasets</a> · <a href="#api-usage">API usage</a> · <a href="#quick-start">Quick start</a> · <a href="#onnx-sdk">ONNX &amp; SDKs</a> · <a href="#demos">Demos</a> · <a href="#results">Evaluation</a> · <a href="#community-projects">Community</a> · <a href="https://valen-team.github.io/cookbook/api-quickstart/">Cookbook</a>
</p>

<a id="version-selection"></a>

## Choose a version

For a first try, use the [web experience or free API](https://valen-team.github.io/) without downloading weights. [Which version should I choose?](https://valen-team.github.io/docs/valen/choosing/) · [GGUF local service](https://valen-team.github.io/docs/valen/gguf/)

<a id="introduction"></a>

## Introduction

Valen (万澜) brings visual perception to System One decision-making. Inspired by [Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev), it evaluates text, images and video against task instructions and returns probabilities over supplied candidates, giving software a structured decision interface. The repository includes the model implementation, data processing, SFT and experimental RLCD training, and inference and evaluation commands, with support for training on your own data.

<a id="news"></a>

## News

- **2026-10-26**: Qualified F32/Q8_0 GGUFs released in all three sizes; use the [Valen custom CPU runtime](https://valen-team.github.io/docs/valen/gguf/). Unmodified llama.cpp/Ollama support is not claimed.

- **2026-10-25**: Updated the modular ONNX models for [0.8B](https://huggingface.co/Valen-Team/Valen-0.8B-ONNX), [2B](https://huggingface.co/Valen-Team/Valen-2B-ONNX) and [4B](https://huggingface.co/Valen-Team/Valen-4B-ONNX), with [Python](https://github.com/Valen-Team/valen-sdk-python) and [JavaScript / TypeScript](https://github.com/Valen-Team/valen-sdk-js) SDKs and a [C++ text integration](https://github.com/Valen-Team/valen-sdk-cpp). See the [ONNX & SDK guide](https://valen-team.github.io/docs/valen/onnx/).

- **2026-10-21**: The [web chat](https://valen-team.github.io/) and [bilingual documentation](https://valen-team.github.io/docs/) are live, with a [free decision API](#api-usage) requiring no key.
- **2026-10-18**: Released [JevalKit](https://github.com/Valen-Team/JevalKit), an open-source evaluation toolkit for Jev-like decision models, providing unified model interfaces and prepared evaluation datasets. The [🤗 JevalKit Leaderboard](https://huggingface.co/spaces/Valen-Team/JevalKit-Leaderboard).
- **2026-10-13**: Released ONNX and GGUF versions of Valen models, available on [Hugging Face](https://huggingface.co/Valen-Team).
- **2026-10-13**: Added API access to `valen-preview-0923`, `valen-0-8b`, `valen-2b` and `valen-4b`. See the [API basics tutorial](https://valen-team.github.io/cookbook/api-quickstart/) for usage examples.
- **2026-10-07**: **Faster, more accurate, more versatile.** Released the latest multimodal System One model **Valen ([0.8B](https://huggingface.co/Valen-Team/Valen-0.8B), [2B](https://huggingface.co/Valen-Team/Valen-2B) and [4B](https://huggingface.co/Valen-Team/Valen-4B))**, supporting text, images and video, and the [**VisualDecisionBench**](https://huggingface.co/datasets/Valen-Team/VisualDecisionBench) benchmark.
- **2026-09-30**: **Training&Inference Support:** Added Flash Attention 2 support, enabled training across multiple nodes and GPUs, and improved GPU memory utilization during training.
- **2026-09-23**: Open-sourced the **Valen** repository and released [**Valen-preview-0923**](https://huggingface.co/Valen-Team/Valen-Preview-0923) with its corresponding [**training data**](https://huggingface.co/datasets/Valen-Team/Valen-Training-General-100k).

<a id="model-downloads"></a>

## Models and datasets

| Model | Native weights / Transformers | ONNX Runtime | GGUF / custom runtime |
| --- | --- | --- | --- |
| Valen 0.8B | [🤗 Valen-0.8B](https://huggingface.co/Valen-Team/Valen-0.8B) | [🤗 Valen-0.8B-ONNX](https://huggingface.co/Valen-Team/Valen-0.8B-ONNX) | [Valen-0.8B-GGUF](https://huggingface.co/Valen-Team/Valen-0.8B-GGUF) |
| Valen 2B | [🤗 Valen-2B](https://huggingface.co/Valen-Team/Valen-2B) | [🤗 Valen-2B-ONNX](https://huggingface.co/Valen-Team/Valen-2B-ONNX) | [Valen-2B-GGUF](https://huggingface.co/Valen-Team/Valen-2B-GGUF) |
| Valen 4B | [🤗 Valen-4B](https://huggingface.co/Valen-Team/Valen-4B) | [🤗 Valen-4B-ONNX](https://huggingface.co/Valen-Team/Valen-4B-ONNX) | [Valen-4B-GGUF](https://huggingface.co/Valen-Team/Valen-4B-GGUF) |
| Valen-preview-0923 (earlier release) | [🤗 Hugging Face](https://huggingface.co/Valen-Team/Valen-Preview-0923) | — | — |

[VisualDecisionBench](https://huggingface.co/datasets/Valen-Team/VisualDecisionBench) contains Image and Video subsets for visual decision evaluation. Earlier releases include [General 100k](https://huggingface.co/datasets/Valen-Team/Valen-Training-General-100k) for training, [General 5k](https://huggingface.co/datasets/Valen-Team/Valen-Eval-General-5k) for evaluation, and [Sokoban](https://huggingface.co/datasets/Valen-Team/Valen-Eval-Game) for game training and evaluation.

<a id="demos"></a>

## Demos

**Valen** image and video decision-making — click the preview to watch the full demo.

<p align="center">
  <a href="assets/demos/Valen.mp4"><img src="assets/demos/valen-demo-preview.gif" alt="Valen demo preview. Click to watch the full video." width="1000"></a>
</p>

**Game decision-making** — five successful Valen rollouts. Click any preview to watch the full video.

<table width="100%">
  <tr>
    <td colspan="2" width="33%" align="center">
      <a href="assets/demos/games/valen-sokoban.mp4"><img src="assets/demos/games/valen-sokoban-preview.gif" alt="Valen solves a Sokoban puzzle." width="100%"></a><br>
      <b>Sokoban</b><br><sub>Plans a 13-step optimal solution.</sub>
    </td>
    <td colspan="2" width="33%" align="center">
      <a href="assets/demos/games/valen-frozen-lake.mp4"><img src="assets/demos/games/valen-frozen-lake-preview.gif" alt="Valen navigates Frozen Lake." width="100%"></a><br>
      <b>Frozen Lake</b><br><sub>11-step optimal safe route.</sub>
    </td>
    <td colspan="2" width="33%" align="center">
      <a href="assets/demos/games/valen-maze.mp4"><img src="assets/demos/games/valen-maze-preview.gif" alt="Valen solves a maze." width="100%"></a><br>
      <b>Maze</b><br><sub>Completes a 22-step long-horizon route.</sub>
    </td>
  </tr>
  <tr>
    <td colspan="3" width="50%" align="center">
      <a href="assets/demos/games/valen-lane-racer.mp4"><img src="assets/demos/games/valen-lane-racer-preview.gif" alt="Valen completes Lane Racer." width="62%"></a><br>
      <b>Lane Racer</b><br><sub>Changes lanes and clears the course in 10 steps.</sub>
    </td>
    <td colspan="3" width="50%" align="center">
      <a href="assets/demos/games/valen-parkour.mp4"><img src="assets/demos/games/valen-parkour-preview.gif" alt="Valen completes Parkour Runner." width="62%"></a><br>
      <b>Parkour Runner</b><br><sub>Chains jumps, dashes, ducks and runs over 14 steps.</sub>
    </td>
  </tr>
</table>

<a id="results"></a>

## Evaluation results — Valen

[JevalKit](https://github.com/Valen-Team/JevalKit) is our evaluation toolkit for decision models, with unified model interfaces, benchmark integration, and scoring protocols. Use it to evaluate and compare decision models across supported benchmarks.

Valen is also integrated into the [Jev Decision Index](https://huggingface.co/spaces/multimodalart/jev-decision-index). Visit the index for its evaluation results and comparisons with other decision models.

**Visual decisions at 0.8B, 2B and 4B.** Valen uses Qwen3.5 + Mixer with shared-state inference, trained with SFT on millions of samples. The evaluation compares these checkpoints with published decision models on **VisualDecisionBench-Image, VisualDecisionBench-Video and public JevBench**.

<p align="center">
  <a href="assets/figures/evaluation-20261006/eval_v1.png"><img src="assets/figures/evaluation-20261006/eval_v1.png" alt="Valen VisualDecisionBench-Image accuracy by model size: 0.8B 75.94%, 2B 80.41%, 4B 83.06%, compared with published baselines." width="1400"></a>
</p>

<p align="center">
  <a href="assets/figures/evaluation-20261006/video.png"><img src="assets/figures/evaluation-20261006/video.png" alt="Valen VisualDecisionBench-Video accuracy by model size: 0.8B 79.03%, 2B 82.85%, 4B 87.36%, compared with published baselines." width="1400"></a>
</p>

<p align="center">
  <a href="assets/figures/evaluation-20261006/jevbench.png"><img src="assets/figures/evaluation-20261006/jevbench.png" alt="Valen public JevBench accuracy: 0.8B 77.92%, 2B 83.55%, 4B 83.12%. InternDecision 4B scores 86.58%." width="1400"></a>
</p>

### Choose what to automate

Use Valen’s probabilities to choose which tickets to route automatically and which need human review. [Compare 0.8B, 2B and 4B](https://valen-team.github.io/docs/valen/reliability/) by measured automated coverage and error rate, then [calibrate and save a routing rule](https://valen-team.github.io/cookbook/calibrated-routing/) for your own data without retraining the model.

**What if no option fits?** A customer might ask to change an address when the choices only cover refunds, delivery tracking and invoices. Add a “none of these” choice and route its selections for review. In a test of 200 English tickets, incorrect automatic acceptance of out-of-scope requests fell from 8 to 1, while 3 more in-scope requests went to review. Validate the routing policy on your own data. [See the complete example and runnable code](https://valen-team.github.io/cookbook/unknown-categories/).

[Try Valen online →](https://valen-team.github.io/?example=ticket)

Earlier results remain in the [archived README](history/README_preview.md).

<a id="quick-start"></a>

## Quick start

For a local HTTP service, start with the [deployment guide](https://valen-team.github.io/docs/valen/onnx/). Complete, tested application tutorials are maintained only in the [official Cookbook](https://valen-team.github.io/cookbook/): [API basics](https://valen-team.github.io/cookbook/api-quickstart/), [local inference](https://valen-team.github.io/cookbook/local-inference/), ticket routing, [when no option fits](https://valen-team.github.io/cookbook/unknown-categories/), product-image screening and video tagging. [简体中文](https://valen-team.github.io/zh-Hans/cookbook/).

<a id="api-usage"></a>

For hosted API data use and retention, see [API data handling and retention](https://valen-team.github.io/docs/valen/api/data-handling/).

### 1. API usage

No model download or GPU is required. Try the [web chat](https://valen-team.github.io/), or call the free endpoint from Python:

```bash
python -m pip install requests
```

```python
import requests

response = requests.post(
    "https://api.valen-team.workers.dev/v1/decisions",
    headers={"Origin": "https://valen-team.github.io"},
    json={
        "model": "valen-2b",
        "input": "Please send my order to my new address.",
        "questions": [{
            "name": "q0",
            "type": "predicate",
            "instructions": "Is the user requesting a delivery address change?",
        }],
        "temperature": 1.0,
    },
    timeout=120,
)
response.raise_for_status()
answer = response.json()["answers"][0]
print(answer["probability"])  # Probability the condition is true, from 0 to 1
```

#### Three native output types

| Type | Question fields | Main answer fields |
| --- | --- | --- |
| `predicate` | `instructions` | `probability`: probability the condition is true |
| `choice` | `instructions`, `choices: [{value, description}, …]` | `choice` and candidate `probabilities` |
| `score` | `instructions`, `levels: [{label, description}, …]` | `score` and level `probabilities` |

For a choice question, replace `questions[0]` in the example with:

```json
{
  "name": "q0",
  "type": "choice",
  "instructions": "Which part of the order does the user want to change?",
  "choices": [
    {"value": "address", "description": "Delivery address"},
    {"value": "quantity", "description": "Item quantity"}
  ]
}
```

For a score, supply ordered levels:

```json
{
  "name": "q0",
  "type": "score",
  "instructions": "Rate the user's intent to change the order.",
  "levels": [
    {"label": "Low", "description": "No request to change the order"},
    {"label": "Medium", "description": "Asks whether a change is possible"},
    {"label": "High", "description": "Explicitly requests a change"}
  ]
}
```

`score` is the probability-weighted average of level indices starting at 0: the three levels above span 0–2; five levels span 0–4. Score distribution entries contain `label`, numeric index `value`, and `probability`. `confidence` is separate from the score and the maximum candidate probability.

#### Free access and higher quotas

Use [List Models](https://valen-team.github.io/docs/valen/api/list-models/) to choose a model ID. Request formats and image examples are in the [API reference](https://valen-team.github.io/docs/valen/api/).

For batch workloads or higher quotas, email [liuziyu77@sjtu.edu.cn](mailto:liuziyu77@sjtu.edu.cn) with your use case and expected volume. Once approved, use the official endpoint and your own key:

```python
import os

url = "https://llm-proxy.openxxlab.com/v1/decisions"
headers = {"Authorization": "Bearer " + os.environ["VALEN_API_KEY"]}
# Replace the URL and headers in requests.post above; keep the same payload shape.
```

Keep keys in server-side environment variables.

### 2. Local inference

| Goal | Starting point |
| --- | --- |
| C++ Linux CPU HTTP / GGUF | [Valen custom llama-server](https://valen-team.github.io/docs/valen/gguf/) |
| Python CPU / NVIDIA CUDA, Node.js or browser applications | [ONNX & SDKs](https://valen-team.github.io/docs/valen/onnx/) |
| Native PyTorch / Transformers model | [Transformers](#transformers) |
| GPU HTTP service | [SGLang](serving/sglang/README.md) · [vLLM](serving/vllm/README.md) |
| Native C++ text integration | [C++ repository](https://github.com/Valen-Team/valen-sdk-cpp) |


#### SGLang image

Use Linux x86-64, an NVIDIA GPU supporting FA3 (such as H100 or H200), a compatible CUDA driver, and Docker with Buildx and NVIDIA Container Toolkit. Build and start the image from the Valen repository root:

```bash
docker buildx build --platform linux/amd64 --load \
  -t valen-sglang:0.1.0-dev3 \
  -f serving/sglang/docker/Dockerfile .
docker run --rm --gpus device=0 --ipc=host \
  -p 127.0.0.1:30000:30000 \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  valen-sglang:0.1.0-dev3 \
  --model-path Valen-Team/Valen-2B --served-model-name valen \
  --host 0.0.0.0 --port 30000
```

#### SGLang plugin

For a compatible official SGLang environment, install the `valen-sglang` wheel. See the [plugin deployment guide](serving/sglang/README.md).

Both deployment options expose `/v1/systemone` and `/v1/decisions` through the same native Valen decision head. They support single-GPU text/image predicate, choice, and score questions. Once the server is ready:

```bash
curl http://localhost:30000/v1/decisions \
  -H 'Content-Type: application/json' \
  -d '{"model":"valen","input":"A cat is on the sofa.","questions":[{"name":"on_sofa","type":"predicate","instructions":"The cat is on the sofa."}]}'
```

To select SystemOne, express the same question as:

```bash
curl http://localhost:30000/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{"model":"valen","state":"A cat is on the sofa.","questions":{"on_sofa":{"type":"noul","instructions":"The cat is on the sofa."}}}'
```


Full instructions: [Image deployment](https://valen-team.github.io/docs/valen/sglang-image/) · [Plugin deployment](https://valen-team.github.io/docs/valen/sglang-plugin/).

#### vLLM image

Linux x86-64, an NVIDIA GPU, a compatible CUDA driver, and Docker with Buildx and NVIDIA Container Toolkit. Build the image from the repository root:

```bash
docker buildx build --platform linux/amd64 --load \
  -t valen-vllm:0.1.0-dev2 \
  -f serving/vllm/docker/Dockerfile .
```

```bash
docker run --rm --gpus device=0 --ipc=host \
  -p 127.0.0.1:8000:8000 \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  valen-vllm:0.1.0-dev2 \
  --model-path Valen-Team/Valen-2B \
  --served-model-name valen \
  --host 0.0.0.0 --port 8000
```

#### vLLM plugin

Install the plugin in the [vLLM environment](serving/vllm/README.md) and start the service:

```bash
bash /absolute/path/to/serving-kit/vllm/install.sh
MODEL=/absolute/path/to/Valen-2B MODEL_NAME=valen CUDA_VISIBLE_DEVICES=0 \
  bash /absolute/path/to/serving-kit/vllm/serve.sh
```

The service listens at `127.0.0.1:8000` and supports `/v1/systemone` and `/v1/decisions`. Full instructions: [Image deployment](https://valen-team.github.io/docs/valen/vllm-image/) · [Plugin deployment](https://valen-team.github.io/docs/valen/vllm-plugin/).

#### Transformers

Install the dependencies and load a model:

```bash
python -m pip install torch torchvision transformers pillow av
```

```python
import torch
from transformers import AutoModel

model = (
    AutoModel.from_pretrained(
        "Valen-Team/Valen-2B",
        trust_remote_code=True,
        dtype="auto",
        attn_implementation="sdpa",
    )
    .to("cuda")
    .eval()
)
torch.set_float32_matmul_precision("highest")
torch.backends.cudnn.allow_tf32 = False

answer = model.predict(
    {
        "state": "A cat is sitting on the sofa.",
        "questions": {
            "animal": {
                "type": "choice",
                "instructions": "Which animal is present?",
                "criteria": {"cat": "A cat", "dog": "A dog"},
            },
            "on_sofa": {"type": "noul", "instructions": "The cat is on the sofa."},
            "clarity": {
                "type": "score",
                "instructions": "How clear is the evidence?",
                "criteria": ["Unclear", "Partly clear", "Clear"],
            },
        },
    },
    execution="shared_state",
    temperature=1.0,
)
print(answer["answers"]["animal"]["choice"])
print(answer["answers"]["animal"]["probabilities"])
print(answer["answers"]["on_sofa"]["noul"])
print(answer["answers"]["clarity"]["score"])
```

Send a Decisions request to the same model:

```python
answer = model.predict(
    {
        "input": "A cat is sitting on the sofa.",
        "questions": [
            {
                "name": "on_sofa",
                "type": "predicate",
                "instructions": "The cat is on the sofa.",
            }
        ],
    },
    execution="shared_state",
    temperature=1.0,
)
print(answer["answers"])
```

Both protocols support text, images, video, and `shared_state` / `question` execution. See [Transformers inference](https://valen-team.github.io/docs/valen/quickstart/) for complete examples.

The first load caches native model weights. Once cached, add `local_files_only=True` to `AutoModel.from_pretrained(...)` to prevent downloads, or pass a local model directory. Use native `Valen-*B` weights here; load ONNX models through the SDKs below.

<a id="onnx-sdk"></a>

#### ONNX and SDKs

Run inference in your application with ONNX Runtime, without installing PyTorch or Transformers. Python supports CPU / NVIDIA CUDA, JavaScript / TypeScript supports Node.js CPU and browser WebGPU, and C++ currently provides a SystemOne text FP32 CPU integration.

Local Python inference needs Python 3.11+; HTTP-only clients support Python 3.10+. Python downloads the full selected variant on first use, such as about 3.505 GB for 0.8B FP32, and reuses the cache on later runs. A local model directory also works.

```bash
python -m pip install "valen-sdk[onnx]==0.2.0"
```

```python
from valen_sdk import Valen

with Valen.local("Valen-Team/Valen-0.8B-ONNX", variant="fp32") as client:
    result = client.decisions(
        input="Please send my order to my new address.",
        questions=[{"name": "change_address", "type": "predicate",
                    "instructions": "Is the customer requesting a delivery address change?"}],
    )
    print(result.by_name("change_address").probability)
```

The example prints a probability from 0 to 1; a higher value means the model is more confident that the customer is requesting a delivery address change.

Use `q8-mlp` to reduce download size and text memory, `cuda-fp16` for CUDA, and the dedicated 0.8B `web-portable` variant for browsers.

See the [ONNX & SDK guide](https://valen-team.github.io/docs/valen/onnx/) for images/video, SystemOne, CUDA, browser inference, the local HTTP server and offline caching. [Python SDK](https://github.com/Valen-Team/valen-sdk-python) · [JavaScript SDK](https://github.com/Valen-Team/valen-sdk-js) · [C++ text integration](https://github.com/Valen-Team/valen-sdk-cpp). To export your own checkpoint, use the [modular ONNX exporter](scripts/export/qwen/README-onnx-v2.md).

### 3. Evaluate VisualDecisionBench

Download and unpack the media, then run the [evaluation script](evaluation/visualdecisionbench/evaluate.py):

```bash
hf download Valen-Team/VisualDecisionBench --repo-type dataset --local-dir data/VisualDecisionBench
python data/VisualDecisionBench/unpack_assets.py
python -m evaluation.visualdecisionbench.evaluate \
  --model Valen-Team/Valen-2B --data data/VisualDecisionBench \
  --output output/visualdecisionbench
```

Add `--lora --revision lora-ckpt` to evaluate the earlier LoRA model. Outputs: `predictions.jsonl` and `metrics.json`, including image/video and Choice/Noul/Score breakdowns and elapsed time. Videos use 16 frames; accuracy uses hard labels, while soft-label Score questions contribute probability metrics.

### 4. Training and fine-tuning

Start with complete Valen-0.8B and LoRA SFT. Head warmup / joint SFT from Qwen is not language-model pretraining. RLCD is experimental.

[Methods and configuration](https://valen-team.github.io/docs/valen/training/) · [complete executable domain recipe](https://valen-team.github.io/cookbook/domain-finetuning/) · [save, resume and export](https://valen-team.github.io/docs/valen/checkpoints/)

<a id="community-projects"></a>

## Community projects

Selected public projects that integrate Valen models or use Valen datasets. Descriptions reflect the linked sources; inclusion does not imply affiliation, endorsement, or independent reproduction. Check each repository's license and supported model version before use.

| Project | Connection to Valen | Evidence and scope |
| --- | --- | --- |
| [Jev-ToM](https://github.com/HKUST-KnowComp/Jev-ToM) | Research evaluation | Evaluates Valen-Preview-0923 on Theory of Mind benchmarks and cites its model card in an [arXiv preprint](https://arxiv.org/abs/2610.18610); results apply to that checkpoint and the paper's evaluation settings. |
| [vLLM Jev](https://github.com/mode-io/vllm-jev) | Model serving | Serves Valen-Preview-0923 for text, image and short-video decisions on documented Linux and Mac paths; see the [model matrix](https://github.com/mode-io/vllm-jev#supported-models). |
| [System1-Omni](https://github.com/InferMatrixLab/system1-omni) | Reference serving | [Python worker](https://github.com/InferMatrixLab/system1-omni/blob/main/recipe/valen/reference.md) for Valen-Preview-0923, text and one image; no video or native Valen CUDA executor in this path. No repository license was found in GitHub metadata. |
| [JevLight](https://github.com/usail-hkust/JevLight) | Simulation configuration | Traffic-signal simulation includes [Valen model configurations](https://github.com/usail-hkust/JevLight/tree/main/config/valen-2b); this establishes a configured integration, not completed results. No repository license was found in GitHub metadata. |
| [reflex](https://github.com/ptlnextdoor/reflex) | Model experiments | [Valen-0.8B reference and MLX experiments](https://github.com/ptlnextdoor/reflex/blob/main/bench/infer/RESULTS.md) on synthetic game frames; author reports limitations. No repository license was found in GitHub metadata. |
| [LLM2Jev](https://github.com/Yinsongxu/LLM2Jev) | Dataset evaluation | Uses [Valen-Eval-Game](https://github.com/Yinsongxu/LLM2Jev#valen-eval-game) for a single-step Sokoban evaluation of its own inference pipeline. |
| [tasksource](https://github.com/sileod/tasksource) | Dataset adaptation | Repackages Valen game examples into typed decisions; see its [data audit](https://github.com/sileod/tasksource/blob/main/docs/jev/audits/game-expansion-audit.md) for split and label details. |

<a id="contributions"></a>

## Contributions

Contributions to Valen are welcome. Open an [issue](https://github.com/Valen-Team/Valen/issues) to report a problem, share a use case or discuss experimental results. Submit a [pull request](https://github.com/Valen-Team/Valen/pulls) to improve the code or documentation, contribute training data or add evaluation tasks. Active contributors can become part of Valen's core development team.

Scan the QR code below to join the Valen WeChat group, discuss the project and share your experiments.

<p align="center">
  <img src="assets/figures/wechat_1021.png" alt="QR code for the Valen WeChat discussion group" width="200">
</p>

<a id="license-and-acknowledgments"></a>

## License and acknowledgments

The code is released under [Apache 2.0](LICENSE). Base models and source datasets retain their respective licenses.

Built on [Qwen3.5](https://huggingface.co/Qwen/Qwen3.5-2B), with the decision interface inspired by [TypeSafe's Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev).

<a id="citation"></a>

## Citation

If you use Valen in your work, please cite this repository:

```bibtex
@misc{valen,
  author       = {{Valen Team}},
  title        = {Valen},
  howpublished = {GitHub - Valen-Team/Valen: A multimodal System One decision model for text, images and video.},
  note         = {GitHub repository},
  year         = {2026},
  url          = {https://github.com/Valen-Team/Valen}
}
```

User guides are maintained only on the website: [Documentation](https://valen-team.github.io/docs/) · [Fine-tuning](https://valen-team.github.io/docs/valen/training/) · [Probabilities](https://valen-team.github.io/docs/valen/probabilities/) · [Cookbook](https://valen-team.github.io/cookbook/).
