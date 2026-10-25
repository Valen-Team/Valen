# ONNX 推理与 SDK

[English](onnx.md) · **简体中文**

通过 ONNX Runtime 将 Valen 决策接入自己的应用。SDK 使用已发布的 Valen ONNX 组件，包含训练后的视觉主干与 Mixer 决策头；本地 SDK 推理无需安装 PyTorch 或 Transformers。

原生 Python 模型可使用 [Transformers](../README_zh.md#transformers)，本地 ONNX 推理可选择下方 SDK。两条路径均支持 SystemOne 和 Decisions；C++ 当前提供 SystemOne 文本集成示例。

第一次在本地体验时，可从下方 Python CPU 示例开始：选择 0.8B、`fp32` 和 Decisions 协议。ONNX 是模型的运行格式，SDK 负责加载模型、处理输入并返回结果；使用已发布模型时无需自己导出。Transformers 与 ONNX 是两条独立的推理路径，按自己的运行环境选择一条即可。

| 接入方式 | 本地推理 | 仓库 |
| --- | --- | --- |
| Python | CPU 或 NVIDIA CUDA 上的文本、图片与视频；本地 HTTP 服务及同步/异步 HTTP 客户端 | [Python SDK](https://github.com/Valen-Team/valen-sdk-python) |
| JavaScript / TypeScript | Node.js CPU 或浏览器 WebGPU 上的文本、图片与视频；HTTP 客户端 | [JavaScript SDK](https://github.com/Valen-Team/valen-sdk-js) |
| C++ | SystemOne 文本、FP32 CPU、每个进程处理一次请求 | [C++ 文本集成](https://github.com/Valen-Team/valen-sdk-cpp) |

## Python CPU 快速开始

需要 Python 3.11+ 和 Git。

首次使用会下载并缓存所选模型组件。Python SDK 会下载所选变体的完整文件，包含视觉权重；0.8B FP32 约为 3.505 GB。Node 的纯文本请求仅下载需要的组件。离线使用时可指定本地目录或已有缓存。

在新终端中执行：

```bash
git clone https://github.com/Valen-Team/valen-sdk-python.git
cd valen-sdk-python
python -m pip install ".[onnx]"
python examples/local_decisions.py
```

首个示例针对同一段客户消息提出三个问题：

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

0.8B FP32 的示例输出：

```text
Intent: address
Address change probability: 0.998
Explicitness score (0–2): 1.921
```

类别可用于业务路由，0–1 概率可用于筛选需复核的请求，评分可用于排序。评分是从零开始的等级索引的概率加权平均值，因此示例中三个等级对应 0–2。不同精度下数值可能有所变化。

## 模型与变体

| 模型 | `fp32` · CPU | `q8-mlp` · CPU | `cuda-fp16` · Python CUDA | `web-portable` · 浏览器 |
| --- | --- | --- | --- | --- |
| [Valen 0.8B](https://huggingface.co/Valen-Team/Valen-0.8B-ONNX) | 3.505 GB | 2.746 GB | 2.000 GB | 2.443 GB |
| [Valen 2B](https://huggingface.co/Valen-Team/Valen-2B-ONNX) | 8.948 GB | 6.343 GB | 5.184 GB | — |
| [Valen 4B](https://huggingface.co/Valen-Team/Valen-4B-ONNX) | 18.253 GB | 11.741 GB | 9.841 GB | — |

表中为发布清单中的十进制 GB 下载体积，包含视觉组件，不是运行时内存需求。实测内存、延迟与质量对比见模型卡片。

CPU 建议从 `fp32` 开始。`q8-mlp` 缩小 MLP 权重并降低实测文本内存占用，但在已测 CPU 上比 FP32 更慢。专用的 `web-portable` 变体仅提供 0.8B。CPU/CUDA 图使用 ONNX Runtime contrib 算子，不能直接替代浏览器图。

### CUDA

在独立 Python 环境中，从 Python SDK 仓库根目录安装 GPU 依赖：

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

### 缓存与离线使用

所选变体已缓存后，可以禁止网络下载：

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

也可指定本地模型目录：`Valen.local("./models/Valen-0.8B-ONNX", variant="fp32", local_files_only=True)`。请保留清单、图与外部权重文件的原始目录结构。Node 中对应的选项名为 `cacheDir` 与 `localFilesOnly`。

## 图片与视频

在 Python SDK 仓库根目录，传入自己的本地媒体文件：

```bash
python examples/media.py /path/to/photo.jpg
python -m pip install ".[onnx,video]"
python examples/media.py /path/to/clip.mp4 --video
```

示例使用 Decisions 消息中的 `input_image` / `input_video`，视频采样四帧。完整代码见[媒体示例](https://github.com/Valen-Team/valen-sdk-python/blob/main/examples/media.py)。JavaScript SDK 也提供图片/视频示例；Node 视频需在系统中安装 FFmpeg 和 ffprobe。浏览器视频使用 WebCodecs，可用编码取决于浏览器支持情况。

## Node.js 与 TypeScript

需要 Node.js 22.12+ 或 24，以及 Git：

```bash
git clone https://github.com/Valen-Team/valen-sdk-js.git
cd valen-sdk-js
npm ci --ignore-scripts
node examples/local.mjs
```

示例在 CPU 上执行相同的三个决策，SDK 包内提供 TypeScript 类型。安装到自己的应用、本地媒体与 `client.systemOne(...)` 的用法见 [JavaScript README](https://github.com/Valen-Team/valen-sdk-js)。需要 CUDA 时，可运行 Python 服务并从 Node 通过 HTTP 调用。

## 浏览器 WebGPU

浏览器示例可以独立使用。在新终端中克隆 JavaScript SDK 并进入示例目录；如果已经克隆，可直接进入 `valen-sdk-js/examples/browser`：

```bash
git clone https://github.com/Valen-Team/valen-sdk-js.git
cd valen-sdk-js/examples/browser
npm install
npm run dev
```

打开终端显示的 localhost 地址，点击 **Run locally**。示例使用已发布的 0.8B `web-portable` 模型，包含运行时资源、加载反馈与错误提示。首次使用会下载并缓存模型组件，完整变体约为 2.443 GB。

在 `examples/browser` 中运行 `npm run build` 和 `npm run preview` 可预览生产构建。部署需要 HTTPS 或 localhost、支持 WebGPU 的浏览器，以及示例中的 COOP/COEP 响应头。详见[浏览器示例](https://github.com/Valen-Team/valen-sdk-js/tree/main/examples/browser)。

## 协议与本地 HTTP 服务

Decisions 使用问题列表，是非判断返回 `probability`；SystemOne 使用以 ID 为键的问题字典，是非判断返回 `noul`。Python 方法为 `client.decisions(...)` 与 `client.system_one(...)`；JavaScript 为 `client.decisions(...)` 与 `client.systemOne(...)`。两种协议执行同一个模型，各仓库均提供可独立运行的 SystemOne 示例。

在 Python SDK 仓库根目录，启动本地 ONNX 服务：

```bash
python -m pip install ".[onnx,server]"
valen serve --model Valen-Team/Valen-0.8B-ONNX --model-name valen --port 8000
```

服务监听 `http://127.0.0.1:8000`，提供 `/v1/decisions` 与 `/v1/systemone`。在另一个终端运行 `python examples/http_client.py` 即可调用。应用中使用 `Valen.remote("http://127.0.0.1:8000/v1", model="valen")` 接入；仅 HTTP 客户端无需本地模型权重或 ONNX Runtime，Python HTTP 客户端支持 Python 3.10+。


HTTP 图片/视频需使用内嵌的 base64 data URL；SDK 直接推理示例中的本地路径不能作为 HTTP 媒体地址。视频解码还需安装 `python -m pip install ".[video]"`。先用上方纯文本 HTTP 示例确认服务可用，再加入媒体输入。

SDK 默认 `execution="shared_state"`，在同一请求中共享一次前向计算；`execution="question"` 分别编译各问题。`temperature` 默认为 1。端点结构见 [API 文档](decisions.md)；缓存、线程、媒体与鉴权选项见 SDK README。

## 原生 C++ 文本集成

[C++ 仓库](https://github.com/Valen-Team/valen-sdk-cpp) 提供模型下载脚本、Rust tokenizer/模板组件、CMake 构建步骤及完整的 SystemOne JSON 请求。它在 C++ 进程内直接运行 FP32 CPU 文本推理，每个进程处理一次请求。已实测 macOS ARM 示例；Linux 构建步骤尚未实测。图片、视频、Decisions 或 HTTP 请使用 Python 或 JavaScript。

[原生 Transformers 推理](../README_zh.md#transformers) · [模型下载](../README_zh.md#模型下载) · [Valen 源码](https://github.com/Valen-Team/Valen)
