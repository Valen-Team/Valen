# Which version should I choose?

For a first try, use the [web experience](https://valen-team.github.io/) or [free HTTP API](https://valen-team.github.io/docs/valen/api/). No model download is needed. For offline use, start with 0.8B; evaluate your own task before moving to 2B or 4B.

| Goal | Recommended path |
| --- | --- |
| Online trial | Web experience or HTTP API: the shortest trial path |
| Python / JavaScript applications | HTTP SDK with an existing service; for offline inference, ONNX SDK with 0.8B FP32 and cached local assets |
| C++ local HTTP service | GGUF Q8_0 with the Valen custom llama-server Linux x86-64 CPU prebuilt package; see [GGUF](gguf.md) |
| GPU deployment | Native weights with the repository SGLang/vLLM plugins, or Python ONNX `cuda-fp16`; follow each backend's measured scope |
| Fine-tuning | Complete native Valen weights with the repository LoRA/full SFT recipes; see [training](https://github.com/Valen-Team/Valen#5-fine-tuning) |

0.8B/2B/4B denote model size. Native weights, ONNX and GGUF are file formats. FP32/F32, Q8_0 and ONNX `q8-mlp` denote format-specific precision/quantization variants, not interchangeable names. Transformers, ONNX Runtime, custom llama.cpp/Ollama and SGLang/vLLM are execution backends. SDK `Valen.local` loads ONNX, not GGUF; access a GGUF server through HTTP.

All three GGUF sizes publish qualified F32, Q8_0 and matching vision modules. Q4/Q5 failed the quality gates and are not recommended. Custom llama.cpp/Ollama real-model validation covers Linux CPU, not GPU, Windows or other platforms. File sizes describe downloads/disk usage, not peak RAM. No unmeasured throughput or latency is promised.
