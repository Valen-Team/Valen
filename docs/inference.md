# Local HTTP deployment

Choose a deployment route:

| Route | Environment | Instructions |
| --- | --- | --- |
| Python ONNX SDK | Python 3.11+, CPU | [ONNX HTTP service](#onnx-http) |
| Valen-Team SGLang image | Docker with NVIDIA Container Toolkit | [Image deployment](https://valen-team.github.io/docs/valen/sglang-image/) |
| Stock SGLang plugin | A compatible official SGLang environment | [SGLang plugin kit](../serving/sglang/README.md) |
| Valen-Team vLLM image | Docker with NVIDIA Container Toolkit | [vLLM image](../serving/vllm/docker/README.md) |
| vLLM plugin | A compatible official vLLM environment | [vLLM plugin kit](../serving/vllm/README.md) |

The GPU routes above expose `/v1/systemone` and `/v1/decisions` for single-GPU text/image predicate, choice, and score decisions. For direct video inference, use [Transformers](../cookbook/local_inference.ipynb) or the [ONNX SDKs](onnx.md).

SGLang defaults to port `30000`; vLLM defaults to port `8000`.

<a id="onnx-http"></a>

## ONNX HTTP service

Install the Python SDK in a separate environment with Python 3.11+ and Git; the main repository's training dependencies are not required. These commands run on CPU:

```bash
python -m pip install "valen-sdk[onnx,server]==0.2.0"
valen serve --model Valen-Team/Valen-0.8B-ONNX --model-name valen --port 8000
```

The service listens at `127.0.0.1:8000` by default and exposes `/v1/systemone` and `/v1/decisions` for text, images and video. Python caches the full selected variant on first load. For offline use, replace `--model` with a prepared local model directory.

Connect with `Valen.remote("http://127.0.0.1:8000/v1", model="valen")`; JavaScript provides the same `Valen.remote` factory. HTTP-only clients do not need ONNX Runtime or local model weights. See [ONNX & SDKs](onnx.md) and the [Python SDK](https://github.com/Valen-Team/valen-sdk-python).

HTTP images/video must use inline base64 data URLs; local file paths cannot be used as HTTP media URLs. Video additionally needs `python -m pip install "valen-sdk[video]==0.2.0"`. Direct SDK calls can read local media paths.

## Inference inside an application

To run without an HTTP server, use the [Transformers notebook](../cookbook/local_inference.ipynb) or the [ONNX / SDK guide](onnx.md). Python supports CPU / CUDA, JavaScript supports Node.js CPU and browser WebGPU, and C++ provides a SystemOne text FP32 CPU integration.
