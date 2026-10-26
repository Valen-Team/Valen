# 本地 HTTP 部署

选择一种部署方式：

| 方式 | 环境 | 操作说明 |
| --- | --- | --- |
| Python ONNX SDK | Python 3.11+、CPU | [ONNX HTTP 服务](#onnx-http) |
| Valen-Team SGLang 镜像 | 已配置 NVIDIA Container Toolkit 的 Docker | [镜像部署](https://valen-team.github.io/zh-Hans/docs/valen/sglang-image/) |
| 原版 SGLang 插件 | 兼容的官方 SGLang 环境 | [SGLang 插件部署包](../serving/sglang/README_zh.md) |
| Valen-Team vLLM 镜像 | 已配置 NVIDIA Container Toolkit 的 Docker | [vLLM 镜像](../serving/vllm/docker/README_zh.md) |
| vLLM 插件 | 兼容的官方 vLLM 环境 | [vLLM 插件部署包](../serving/vllm/README_zh.md) |

以上 GPU 部署方式均提供 `/v1/systemone` 和 `/v1/decisions`，支持单 GPU 文本与图片的判断、选择和评分。直接视频推理可使用 [Transformers](../cookbook/local_inference_zh.ipynb) 或 [ONNX SDK](onnx_zh.md)。

SGLang 默认使用 `30000` 端口，vLLM 默认使用 `8000`。

<a id="onnx-http"></a>

## ONNX HTTP 服务

在独立环境中安装 Python SDK；需要 Python 3.11+ 和 Git，无需安装主仓库的训练依赖。以下命令使用 CPU：

```bash
python -m pip install "valen-sdk[onnx,server] @ git+https://github.com/Valen-Team/valen-sdk-python.git@v0.2.0"
valen serve --model Valen-Team/Valen-0.8B-ONNX --model-name valen --port 8000
```

服务默认监听 `127.0.0.1:8000`，提供 `/v1/systemone` 与 `/v1/decisions`，支持文本、图片和视频。Python 首次加载会缓存完整的所选变体；离线时可将 `--model` 换成已准备好的本地模型目录。

应用中使用 `Valen.remote("http://127.0.0.1:8000/v1", model="valen")`；JavaScript 使用同名 `Valen.remote` 工厂。仅 HTTP 客户端不需要 ONNX Runtime 或本地模型权重。详见 [ONNX 与 SDK](onnx_zh.md) 和 [Python SDK](https://github.com/Valen-Team/valen-sdk-python)。

HTTP 图片/视频须使用内嵌 base64 data URL，不能将本地文件路径作为 HTTP 媒体地址。视频还需安装 `python -m pip install "valen-sdk[video] @ git+https://github.com/Valen-Team/valen-sdk-python.git@v0.2.0"`；直接 SDK 调用可以读取本地媒体路径。

## 直接在应用中推理

不启动 HTTP 服务时，可使用 [Transformers Notebook](../cookbook/local_inference_zh.ipynb) 或 [ONNX / SDK 指南](onnx_zh.md)。Python SDK 支持 CPU / CUDA；JavaScript 支持 Node.js CPU 与浏览器 WebGPU；C++ 提供 SystemOne 文本 FP32 CPU 集成。
