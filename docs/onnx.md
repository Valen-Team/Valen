# ONNX inference and SDKs

**English** · [简体中文](onnx_zh.md)

Run Valen decisions in your application with ONNX Runtime. The SDKs use the published Valen ONNX components, including the trained vision tower and Mixer decision head. Local SDK inference does not require PyTorch or Transformers.

Use [Transformers](../README.md#transformers) for the native Python model, or choose an SDK below for local ONNX inference. Both routes support SystemOne and Decisions; C++ currently provides a SystemOne text integration example.

For a first local run, start with the Python CPU example below: use 0.8B, `fp32` and the Decisions protocol. ONNX is the model runtime format; the SDK loads it, processes inputs and returns results. Published models are ready to use without exporting them yourself. Transformers and ONNX are independent inference routes; choose one for your environment.

| Integration | Local inference | Repository |
| --- | --- | --- |
| Python | Text, images and video on CPU or NVIDIA CUDA; local HTTP server and sync/async HTTP clients | [Python SDK](https://github.com/Valen-Team/valen-sdk-python) |
| JavaScript / TypeScript | Text, images and video in Node.js on CPU or a WebGPU browser; HTTP client | [JavaScript SDK](https://github.com/Valen-Team/valen-sdk-js) |
| C++ | SystemOne text, FP32 CPU, one request per process | [C++ text integration](https://github.com/Valen-Team/valen-sdk-cpp) |

## Python CPU quickstart

Use Python 3.11+ and Git.

First use downloads and caches the selected model components. The Python SDK downloads the full selected variant, including vision weights; 0.8B FP32 is about 3.505 GB. Node text requests download only their required components. Choose a local directory or an existing cache when working offline.

From a fresh terminal:

```bash
git clone https://github.com/Valen-Team/valen-sdk-python.git
cd valen-sdk-python
python -m pip install ".[onnx]"
python examples/local_decisions.py
```

The first example asks three questions about the same customer message:

```python
from valen_sdk import Valen

with Valen.local("Valen-Team/Valen-0.8B-ONNX", variant="fp32") as client:
    result = client.decisions(
        input="I moved. Please ship my order to 12 Elm Street instead.",
        questions=[
            {"name": "intent", "type": "choice", "instructions": "What does the customer want?",
             "choices": [{"value": "address", "description": "Change the delivery address"},
                         {"value": "cancel", "description": "Cancel the order"}]},
            {"name": "change_address", "type": "predicate",
             "instructions": "The customer requests a delivery address change."},
            {"name": "explicitness", "type": "score", "instructions": "How explicit is the request?",
             "levels": [{"label": "none", "description": "No request"},
                        {"label": "possible", "description": "Asks whether a change is possible"},
                        {"label": "explicit", "description": "Explicitly requests a change"}]},
        ],
    )
    print("Intent:", result.by_name("intent").choice)
    print(f"Address change probability: {result.by_name('change_address').probability:.3f}")
    print(f"Explicitness score (0–2): {result.by_name('explicitness').score:.3f}")
```

Example output with 0.8B FP32:

```text
Intent: address
Address change probability: 0.998
Explicitness score (0–2): 1.921
```

Use the category for routing, the 0–1 probability to select requests for review, and the score to rank them. A score is the probability-weighted mean of zero-based level indices; these three levels span 0–2. Exact outputs may vary with precision.

## Models and variants

| Model | `fp32` · CPU | `q8-mlp` · CPU | `cuda-fp16` · Python CUDA | `web-portable` · browser |
| --- | --- | --- | --- | --- |
| [Valen 0.8B](https://huggingface.co/Valen-Team/Valen-0.8B-ONNX) | 3.505 GB | 2.746 GB | 2.000 GB | 2.443 GB |
| [Valen 2B](https://huggingface.co/Valen-Team/Valen-2B-ONNX) | 8.948 GB | 6.343 GB | 5.184 GB | — |
| [Valen 4B](https://huggingface.co/Valen-Team/Valen-4B-ONNX) | 18.253 GB | 11.741 GB | 9.841 GB | — |

These are decimal GB download sizes from the published manifests, including vision components. They are not runtime memory requirements. The model cards include measured memory, latency and quality comparisons.

Start with `fp32` on CPU. `q8-mlp` reduces MLP weight size and measured text memory, but was slower than FP32 in the measured CPU runs. The dedicated `web-portable` variant is available for 0.8B. CPU/CUDA graphs contain ONNX Runtime contrib operators and cannot be substituted for the browser graph.

### CUDA

In a separate Python environment, install the GPU dependencies from the Python SDK repository root:

```bash
python -m pip install ".[cuda]"
```

```python
from valen_sdk import Valen

with Valen.local(
    "Valen-Team/Valen-0.8B-ONNX", variant="cuda-fp16",
    providers=[("CUDAExecutionProvider", {"use_tf32": "0"}), "CPUExecutionProvider"],
) as client:
    result = client.decisions(
        input="Please send my order to my new address.",
        questions=[{"name": "change_address", "type": "predicate",
                    "instructions": "Is the customer requesting a delivery address change?"}],
    )
    print(result.by_name("change_address").probability)
```

### Cache and offline use

After the selected variant has been cached, prevent network downloads:

```python
from valen_sdk import Valen

with Valen.local(
    "Valen-Team/Valen-0.8B-ONNX", variant="fp32", local_files_only=True,
) as client:
    result = client.decisions(
        input="Please send my order to my new address.",
        questions=[{"name": "change_address", "type": "predicate",
                    "instructions": "Is the customer requesting a delivery address change?"}],
    )
    print(result.by_name("change_address").probability)
```

A local model directory also works: `Valen.local("./models/Valen-0.8B-ONNX", variant="fp32", local_files_only=True)`. Keep the manifest, graphs and external weight files in their original directory structure. Node uses `cacheDir` and `localFilesOnly` for the corresponding options.

## Images and video

From the Python SDK repository root, use your own local media files:

```bash
python examples/media.py /path/to/photo.jpg
python -m pip install ".[onnx,video]"
python examples/media.py /path/to/clip.mp4 --video
```

The examples use Decisions messages with `input_image` / `input_video` and sample four video frames. See the [complete media example](https://github.com/Valen-Team/valen-sdk-python/blob/main/examples/media.py). The JavaScript SDK provides the same image/video example; Node video needs externally installed FFmpeg and ffprobe. Browser video uses WebCodecs and depends on the browser's codec support.

## Node.js and TypeScript

Use Node.js 22.12+ or 24 and Git:

```bash
git clone https://github.com/Valen-Team/valen-sdk-js.git
cd valen-sdk-js
npm ci --ignore-scripts
node examples/local.mjs
```

The example runs the same three CPU decisions. TypeScript declarations are included. See the [JavaScript README](https://github.com/Valen-Team/valen-sdk-js) for installing the SDK in your own application, local media inputs and `client.systemOne(...)`. For CUDA, run the Python server and call it over HTTP from Node.

## Browser WebGPU

The browser example can be used independently. In a fresh terminal, clone the JavaScript SDK and enter its example directory. If already cloned, go directly to `valen-sdk-js/examples/browser`:

```bash
git clone https://github.com/Valen-Team/valen-sdk-js.git
cd valen-sdk-js/examples/browser
npm install
npm run dev
```

Open the localhost URL and click **Run locally**. The example uses the published 0.8B `web-portable` model and includes runtime assets, loading feedback and error handling. First use downloads and caches model components; the complete variant is about 2.443 GB.

From `examples/browser`, use `npm run build` and `npm run preview` for a production preview. Deployment needs HTTPS or localhost, a WebGPU-capable browser, and the example's COOP/COEP response headers. See the [browser example](https://github.com/Valen-Team/valen-sdk-js/tree/main/examples/browser).

## Protocols and local HTTP service

Decisions uses a question list and exposes predicate results as `probability`. SystemOne uses questions keyed by ID and exposes true/false results as `noul`. The Python methods are `client.decisions(...)` and `client.system_one(...)`; JavaScript uses `client.decisions(...)` and `client.systemOne(...)`. Both execute the same model. Complete independent SystemOne examples are included in each repository.

From the Python SDK repository root, start a local ONNX service:

```bash
python -m pip install ".[onnx,server]"
valen serve --model Valen-Team/Valen-0.8B-ONNX --model-name valen --port 8000
```

The service listens at `http://127.0.0.1:8000` and exposes `/v1/decisions` and `/v1/systemone`. In another terminal, run `python examples/http_client.py`. Use `Valen.remote("http://127.0.0.1:8000/v1", model="valen")` to connect; HTTP-only clients do not need local model weights or ONNX Runtime. Python HTTP clients support Python 3.10+.


HTTP images/video must use inline base64 data URLs; local paths from direct SDK examples cannot be used as HTTP media URLs. Video decoding additionally needs `python -m pip install ".[video]"`. Start with the text-only HTTP example above, then add media inputs.

`execution="shared_state"` is the SDK default and shares a forward pass within a request; `execution="question"` compiles questions separately. `temperature` defaults to 1. See the [API reference](decisions.md) for endpoint schemas and the SDK READMEs for cache, threading, media and authentication options.

## Native C++ text integration

The [C++ repository](https://github.com/Valen-Team/valen-sdk-cpp) includes a model download script, Rust tokenizer/template components, CMake build instructions and a complete SystemOne JSON request. It runs FP32 CPU text inference directly in a C++ process, with one request per process. Its macOS ARM example has been tested; the Linux build instructions have not been tested. Use Python or JavaScript for images, video, Decisions or HTTP.

[Native Transformers inference](../README.md#transformers) · [Model downloads](../README.md#model-downloads) · [Valen source](https://github.com/Valen-Team/Valen)
