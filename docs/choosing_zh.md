# 应该选择哪个版本

首次体验默认使用[在线体验](https://valen-team.github.io/)或[免费 HTTP API](https://valen-team.github.io/zh-Hans/docs/valen/api/)，无需下载模型。需要离线时先用 0.8B，再根据自己的任务评测决定是否换 2B/4B。

| 需求 | 推荐路径 |
| --- | --- |
| 在线体验 | 网页或 HTTP API，最短试用路径 |
| Python / JavaScript 应用 | HTTP SDK 连接已有服务；离线用 ONNX SDK，先选 0.8B FP32，复用本地缓存 |
| C++ 本地 HTTP 服务 | GGUF Q8_0 + Valen 定制 llama-server Linux x86-64 CPU 预编译包，见 [GGUF](gguf_zh.md) |
| GPU 部署 | 原生权重 + 仓库的 SGLang/vLLM 插件，或 Python ONNX `cuda-fp16`；按各后端已验证范围部署 |
| 微调 | 原生完整 Valen 权重 + 主仓库 LoRA/full SFT 配置，见 [训练说明](https://github.com/Valen-Team/Valen#5-fine-tuning) |

0.8B/2B/4B 是模型尺寸，不是文件格式。原生权重、ONNX、GGUF 是不同格式；FP32/F32、Q8_0、ONNX `q8-mlp` 是各格式中的精度/量化变体，不能互换名称。Transformers、ONNX Runtime、定制 llama.cpp/Ollama、SGLang/vLLM 是运行后端。SDK 的 `Valen.local` 加载 ONNX，不加载 GGUF；GGUF 服务可通过 HTTP 调用。

GGUF 三个尺寸只发布通过质量门槛的 F32、Q8_0 和匹配视觉模块。Q4/Q5 未通过，不推荐。定制 llama.cpp/Ollama 的真实模型验证范围为 Linux CPU；不能推广到未验证的 GPU、Windows 或其他平台。文件大小是下载/磁盘大小，不是峰值内存需求；没有提供未经测量的吞吐或延迟承诺。
