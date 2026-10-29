<p align="center">
  <img src="assets/branding/valen-mark-white.png" alt="Valen" width="112">
</p>
<h1 align="center">Valen</h1>
<p align="center"><strong>快速决策，持续进化。</strong></p>
<p align="center">多模态 System One 决策模型。输入文本、图像和视频，输出结构化决策概率。</p>

<p align="center">
  <a href="https://huggingface.co/Valen-Team/Valen-0.8B"><img src="https://img.shields.io/badge/Valen-0.8B-3964FE?style=flat&amp;labelColor=1E232C" alt="Valen 0.8B"></a>
  <a href="https://huggingface.co/Valen-Team/Valen-2B"><img src="https://img.shields.io/badge/Valen-2B-3964FE?style=flat&amp;labelColor=1E232C" alt="Valen 2B"></a>
  <a href="https://huggingface.co/Valen-Team/Valen-4B"><img src="https://img.shields.io/badge/Valen-4B-3964FE?style=flat&amp;labelColor=1E232C" alt="Valen 4B"></a>
  <a href="https://huggingface.co/datasets/Valen-Team/VisualDecisionBench"><img src="https://img.shields.io/badge/Dataset-VisualDecisionBench-3964FE?style=flat&amp;labelColor=1E232C" alt="VisualDecisionBench"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-555555?style=flat&amp;labelColor=1E232C" alt="Apache 2.0"></a>
  <a href="pyproject.toml"><img src="https://img.shields.io/badge/Python-3.10%2B-555555?style=flat&amp;labelColor=1E232C" alt="Python 3.10+"></a>
</p>

<p align="center">
  <a href="README.md">English</a> · <strong>简体中文</strong><br>
  <a href="https://valen-team.github.io/zh-Hans/">在线体验</a> · <a href="https://valen-team.github.io/zh-Hans/docs/">文档</a> · <a href="https://valen-team.github.io/zh-Hans/docs/valen/api/">API</a> · <a href="https://github.com/Valen-Team/skills">Agent skill</a> · <a href="https://huggingface.co/Valen-Team">Hugging Face</a>
</p>

<p align="center">
  <a href="https://github.com/Valen-Team/valen-sdk-python">Python SDK</a> · <a href="https://github.com/Valen-Team/valen-sdk-js">JavaScript SDK</a> · <a href="https://github.com/Valen-Team/valen-sdk-cpp">C++ 文本集成</a>
</p>

<p align="center">
  <a href="#简介">简介</a> · <a href="#模型下载">模型与数据集</a> · <a href="#api-usage">API 调用</a> · <a href="#快速开始">快速开始</a> · <a href="#onnx-sdk">ONNX 与 SDK</a> · <a href="#演示">演示</a> · <a href="#实验结果">评测结果</a> · <a href="#community-projects">社区项目</a> · <a href="https://valen-team.github.io/zh-Hans/cookbook/api-quickstart/">Cookbook</a>
</p>

<a id="简介"></a>

## 简介

Valen（万澜）将视觉感知引入 System One 决策。受 [Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev) 启发，它根据任务指令，对文本、图像和视频中的信息进行判断，直接输出给定候选的概率分布，为程序提供结构化的决策接口。仓库提供模型实现、数据处理、SFT 与实验性 RLCD 训练，以及推理和评估命令，支持使用自有数据训练。

<a id="version-selection"></a>

## 版本选择

首次体验使用[网页或免费 API](https://valen-team.github.io/)，无需下载模型。[应该选择哪个版本？](https://valen-team.github.io/zh-Hans/docs/valen/choosing/) · [GGUF 本地服务](https://valen-team.github.io/zh-Hans/docs/valen/gguf/)

<a id="news"></a>

## 最新动态

- **2026-10-29**：[Python SDK](https://github.com/Valen-Team/valen-sdk-python/blob/main/README_zh.md#与大模型对比) 新增 `Valen.llm()`：把同一套问题交给 GPT、Claude 或 Gemini，返回同样的类型化答案，可在自己的代码里直接对比 Valen 与大模型的决策、token 和延迟。
- **2026-10-29**：发布面向 Claude Code、Cursor、Codex 等编程 agent 的 [Valen agent skill](https://github.com/Valen-Team/skills)，官网同步提供 [`llms.txt`](https://valen-team.github.io/zh-Hans/llms.txt)、每个文档页的 Markdown 版本和 [OpenAPI 规格](https://valen-team.github.io/openapi.json)。[JevalKit](https://github.com/Valen-Team/JevalKit) 新增 OpenAI、Anthropic、Gemini 通用大模型基线，便于直接对比。
- **2026-10-26**: 三尺寸 F32/Q8_0 GGUF 已发布；通过 [Valen 补丁版 llama.cpp 和 Ollama](https://valen-team.github.io/zh-Hans/docs/valen/gguf/) 运行，直接返回候选项概率而不是生成文本。
- **2026-10-25**：更新 [0.8B](https://huggingface.co/Valen-Team/Valen-0.8B-ONNX)、[2B](https://huggingface.co/Valen-Team/Valen-2B-ONNX)、[4B](https://huggingface.co/Valen-Team/Valen-4B-ONNX) 模块化 ONNX 模型，发布 [Python](https://github.com/Valen-Team/valen-sdk-python)、[JavaScript / TypeScript](https://github.com/Valen-Team/valen-sdk-js) SDK 与 [C++ 文本集成](https://github.com/Valen-Team/valen-sdk-cpp)。见 [ONNX 与 SDK 指南](https://valen-team.github.io/zh-Hans/docs/valen/onnx/)。
- **2026-10-21**：官网[聊天](https://valen-team.github.io/zh-Hans/)与[中英文文档](https://valen-team.github.io/zh-Hans/docs/)上线；提供无需 Key 的[免费决策 API](#api-usage)。

更早的动态与各版本代码变更见 [CHANGELOG.md](CHANGELOG.md)。

<a id="模型下载"></a>

## 模型与数据集

| 模型 | 原生权重 / Transformers | ONNX Runtime | GGUF / 定制运行时 |
| --- | --- | --- | --- |
| Valen 0.8B | [🤗 Valen-0.8B](https://huggingface.co/Valen-Team/Valen-0.8B) | [🤗 Valen-0.8B-ONNX](https://huggingface.co/Valen-Team/Valen-0.8B-ONNX) | [Valen-0.8B-GGUF](https://huggingface.co/Valen-Team/Valen-0.8B-GGUF) |
| Valen 2B | [🤗 Valen-2B](https://huggingface.co/Valen-Team/Valen-2B) | [🤗 Valen-2B-ONNX](https://huggingface.co/Valen-Team/Valen-2B-ONNX) | [Valen-2B-GGUF](https://huggingface.co/Valen-Team/Valen-2B-GGUF) |
| Valen 4B | [🤗 Valen-4B](https://huggingface.co/Valen-Team/Valen-4B) | [🤗 Valen-4B-ONNX](https://huggingface.co/Valen-Team/Valen-4B-ONNX) | [Valen-4B-GGUF](https://huggingface.co/Valen-Team/Valen-4B-GGUF) |
| Valen-preview-0923（早期版本） | [🤗 Hugging Face](https://huggingface.co/Valen-Team/Valen-Preview-0923) | — | — |

[VisualDecisionBench](https://huggingface.co/datasets/Valen-Team/VisualDecisionBench) 包含 Image 和 Video 两个子集，用于图像与视频决策评测。此前发布的 [General 100k](https://huggingface.co/datasets/Valen-Team/Valen-Training-General-100k) 用于训练，[General 5k](https://huggingface.co/datasets/Valen-Team/Valen-Eval-General-5k) 用于评测，[Sokoban](https://huggingface.co/datasets/Valen-Team/Valen-Eval-Game) 用于游戏训练与评测。

<a id="演示"></a>

## 演示

**Valen** 图像与视频决策演示，点击预览观看完整视频。

<p align="center">
  <a href="assets/demos/Valen.mp4"><img src="assets/demos/valen-demo-preview.gif" alt="Valen 演示预览，点击观看完整视频。" width="1000"></a>
</p>

**游戏决策演示**——五条由 Valen 成功完成的游戏轨迹。点击任一预览可观看完整视频。

<table width="100%">
  <tr>
    <td colspan="2" width="33%" align="center">
      <a href="assets/demos/games/valen-sokoban.mp4"><img src="assets/demos/games/valen-sokoban-preview.gif" alt="Valen 完成推箱子关卡。" width="100%"></a><br>
      <b>推箱子</b><br><sub>规划 13 步最优解。</sub>
    </td>
    <td colspan="2" width="33%" align="center">
      <a href="assets/demos/games/valen-frozen-lake.mp4"><img src="assets/demos/games/valen-frozen-lake-preview.gif" alt="Valen 完成冰湖导航。" width="100%"></a><br>
      <b>冰湖</b><br><sub>11 步安全最优路线。</sub>
    </td>
    <td colspan="2" width="33%" align="center">
      <a href="assets/demos/games/valen-maze.mp4"><img src="assets/demos/games/valen-maze-preview.gif" alt="Valen 完成迷宫关卡。" width="100%"></a><br>
      <b>迷宫</b><br><sub>完成 22 步长程路线规划。</sub>
    </td>
  </tr>
  <tr>
    <td colspan="3" width="50%" align="center">
      <a href="assets/demos/games/valen-lane-racer.mp4"><img src="assets/demos/games/valen-lane-racer-preview.gif" alt="Valen 完成车道竞速关卡。" width="62%"></a><br>
      <b>车道竞速</b><br><sub>连续变道避障，10 步完成赛道。</sub>
    </td>
    <td colspan="3" width="50%" align="center">
      <a href="assets/demos/games/valen-parkour.mp4"><img src="assets/demos/games/valen-parkour-preview.gif" alt="Valen 完成跑酷关卡。" width="62%"></a><br>
      <b>跑酷</b><br><sub>组合 14 步跳跃、冲刺、下蹲与奔跑动作。</sub>
    </td>
  </tr>
</table>

<a id="实验结果"></a>

## 评测结果 — Valen

[JevalKit](https://github.com/Valen-Team/JevalKit) 是我们的决策模型评测工具，提供统一的模型接口、Benchmark 接入和评分协议，可用于在支持的评测任务上评估与比较不同决策模型。

Valen 也已接入 [Jev Decision Index](https://huggingface.co/spaces/multimodalart/jev-decision-index)，可前往查看该评测中的结果及与其他决策模型的对比。

**覆盖 0.8B、2B、4B 的视觉决策模型。** Valen 采用 Qwen3.5 + Mixer 和共享 state 推理，在百万级样本上进行 SFT。以下在 **VisualDecisionBench-Image、VisualDecisionBench-Video、公开 JevBench** 三组评测中，与已发布的决策模型进行对比。

<p align="center">
  <a href="assets/figures/evaluation-20261006/eval_v1.png"><img src="assets/figures/evaluation-20261006/eval_v1.png" alt="Valen VisualDecisionBench-Image 准确率按模型规模排列：0.8B 75.94%、2B 80.41%、4B 83.06%，与发布基线比较。" width="1400"></a>
</p>

<p align="center">
  <a href="assets/figures/evaluation-20261006/video.png"><img src="assets/figures/evaluation-20261006/video.png" alt="Valen VisualDecisionBench-Video 准确率按模型规模排列：0.8B 79.03%、2B 82.85%、4B 87.36%，与发布基线比较。" width="1400"></a>
</p>

<p align="center">
  <a href="assets/figures/evaluation-20261006/jevbench.png"><img src="assets/figures/evaluation-20261006/jevbench.png" alt="Valen 公开 JevBench 准确率：0.8B 77.92%、2B 83.55%、4B 83.12%；InternDecision 4B 为 86.58%。" width="1400"></a>
</p>

### 决定哪些工作交给 Valen

用 Valen 返回的概率，决定哪些工单自动分流、哪些交给人工复核。[对比三个规模模型的自动处理比例与错误率](https://valen-team.github.io/zh-Hans/docs/valen/reliability/)，再用[完整示例](https://valen-team.github.io/zh-Hans/cookbook/calibrated-routing/)在自己的数据上校准并保存分流规则，无需重新训练模型。

**候选项都不合适怎么办？** 例如，工单要求修改地址，但候选项只有退款、查物流和开发票。加入“都不适用”候选后，可以将选中它的请求转人工。在一项 200 条英文工单测试中，候选外请求被错误自动接收的数量从 8 条降到 1 条，同时多了 3 条正常请求转人工。用自己的数据验证分流效果。[查看完整案例和可运行代码](https://valen-team.github.io/zh-Hans/cookbook/unknown-categories/)。

[在线试用 Valen →](https://valen-team.github.io/zh-Hans/?example=ticket)

早期结果保留在[归档 README](history/README_zh_preview.md)。

<a id="快速开始"></a>

## 快速开始

启动本地 HTTP 服务请阅读[部署指南](https://valen-team.github.io/zh-Hans/docs/valen/onnx/)。完整、经过实测的应用教程只在[官网 Cookbook](https://valen-team.github.io/zh-Hans/cookbook/)维护，包括 [API 入门](https://valen-team.github.io/zh-Hans/cookbook/api-quickstart/)、[本地推理基础](https://valen-team.github.io/zh-Hans/cookbook/local-inference/)、工单分流、[候选项都不对](https://valen-team.github.io/zh-Hans/cookbook/unknown-categories/)、商品图片筛选和短视频标注。[English](https://valen-team.github.io/cookbook/)。

想让编程 agent 帮你完成集成，可以安装 [Valen agent skill](https://github.com/Valen-Team/skills)：

```bash
claude plugin marketplace add Valen-Team/skills && claude plugin install valen@valen-team  # Claude Code
npx skills add Valen-Team/skills --skill valen                                             # 其他 agent
```

<a id="api-usage"></a>

托管 API 的数据用途与保留说明见 [API 数据处理与保留](https://valen-team.github.io/zh-Hans/docs/valen/api/data-handling/)。

### 1. API 调用

无需下载模型或准备 GPU。通过[官网聊天](https://valen-team.github.io/zh-Hans/)直接体验，或在 Python 中调用免费接口：

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
        "input": "请把订单寄到我的新地址。",
        "questions": [{
            "name": "q0",
            "type": "predicate",
            "instructions": "用户是否要求更改收货地址？",
        }],
        "temperature": 1.0,
    },
    timeout=120,
)
response.raise_for_status()
answer = response.json()["answers"][0]
print(answer["probability"])  # 条件成立的概率，范围 0–1
```

#### 三种原生输出

| 类型 | 问题字段 | 主要返回字段 |
| --- | --- | --- |
| `predicate` | `instructions` | `probability`：条件成立的概率 |
| `choice` | `instructions`、`choices: [{value, description}, …]` | `choice` 和候选项 `probabilities` |
| `score` | `instructions`、`levels: [{label, description}, …]` | `score` 和各档位 `probabilities` |

选择题可将上例中的 `questions[0]` 替换为：

```json
{
  "name": "q0",
  "type": "choice",
  "instructions": "用户希望修改订单的哪一项？",
  "choices": [
    {"value": "address", "description": "收货地址"},
    {"value": "quantity", "description": "商品数量"}
  ]
}
```

评分使用有序档位：

```json
{
  "name": "q0",
  "type": "score",
  "instructions": "评估用户修改订单的意愿强度。",
  "levels": [
    {"label": "低", "description": "未提出修改要求"},
    {"label": "中", "description": "询问能否修改"},
    {"label": "高", "description": "明确要求修改"}
  ]
}
```

`score` 是从 0 开始的档位索引的概率加权平均值；上面的三档范围为 0–2，五档则为 0–4。评分分布条目包含 `label`、数值索引 `value` 和 `probability`。`confidence` 是独立字段，不是评分或最大候选概率。

#### 免费额度与更高额度

通过[模型列表](https://valen-team.github.io/zh-Hans/docs/valen/api/list-models/)选择模型 ID。请求格式与图片示例见 [API 文档](https://valen-team.github.io/zh-Hans/docs/valen/api/)。

需要批量调用或更高额度，请邮件联系 [liuziyu77@sjtu.edu.cn](mailto:liuziyu77@sjtu.edu.cn)，说明用途与预计调用量。获批后使用官方接口和自己的 Key：

```python
import os

url = "https://llm-proxy.openxxlab.com/v1/decisions"
headers = {"Authorization": "Bearer " + os.environ["VALEN_API_KEY"]}
# 在上面的 requests.post 中替换 URL 和 headers，其余结构保持一致。
```

Key 只保存在服务端环境变量中。

### 2. 本地推理

| 需求 | 推荐入口 |
| --- | --- |
| C++ Linux CPU HTTP / GGUF | [Valen 定制 llama-server](https://valen-team.github.io/zh-Hans/docs/valen/gguf/) |
| Python CPU / NVIDIA CUDA、Node.js 或浏览器应用 | [ONNX 与 SDK](https://valen-team.github.io/zh-Hans/docs/valen/onnx/) |
| 原生 PyTorch / Transformers 模型 | [Transformers](#transformers) |
| GPU HTTP 服务 | [SGLang](serving/sglang/README_zh.md) · [vLLM](serving/vllm/README_zh.md) |
| 原生 C++ 文本集成 | [C++ 仓库](https://github.com/Valen-Team/valen-sdk-cpp) |


#### SGLang 镜像

需要 Linux x86-64、支持 FA3 的 NVIDIA GPU（如 H100、H200）、兼容的 CUDA 驱动，以及安装 Buildx 和 NVIDIA Container Toolkit 的 Docker。在 Valen 仓库根目录构建镜像并启动：

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

#### SGLang 插件

已有兼容的官方 SGLang 环境时，可安装 `valen-sglang` wheel。完整步骤见[插件部署说明](serving/sglang/README_zh.md)。

两种部署方式均通过 Valen 原生决策头提供 `/v1/systemone` 和 `/v1/decisions`，支持单 GPU 的文本、图片判断、选择与评分。服务就绪后发送请求：

```bash
curl http://localhost:30000/v1/decisions \
  -H 'Content-Type: application/json' \
  -d '{"model":"valen","input":"A cat is on the sofa.","questions":[{"name":"on_sofa","type":"predicate","instructions":"The cat is on the sofa."}]}'
```

选择 SystemOne 协议时，同一问题写为：

```bash
curl http://localhost:30000/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{"model":"valen","state":"A cat is on the sofa.","questions":{"on_sofa":{"type":"noul","instructions":"The cat is on the sofa."}}}'
```


详细步骤：[镜像部署](https://valen-team.github.io/zh-Hans/docs/valen/sglang-image/) · [插件部署](https://valen-team.github.io/zh-Hans/docs/valen/sglang-plugin/)。

#### vLLM 镜像

Linux x86-64、NVIDIA GPU、兼容的 CUDA 驱动，以及安装 Buildx 和 NVIDIA Container Toolkit 的 Docker。在仓库根目录构建镜像：

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

#### vLLM 插件

在[vLLM 环境](serving/vllm/README_zh.md)中安装插件并启动服务：

```bash
bash /absolute/path/to/serving-kit/vllm/install.sh
MODEL=/absolute/path/to/Valen-2B MODEL_NAME=valen CUDA_VISIBLE_DEVICES=0 \
  bash /absolute/path/to/serving-kit/vllm/serve.sh
```

服务地址为 `127.0.0.1:8000`，支持 `/v1/systemone` 和 `/v1/decisions`。完整步骤：[镜像部署](https://valen-team.github.io/zh-Hans/docs/valen/vllm-image/) · [插件部署](https://valen-team.github.io/zh-Hans/docs/valen/vllm-plugin/)。

#### Transformers

安装依赖并加载模型：

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

使用同一个模型发送 Decisions 请求：

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

两种协议均支持文本、图片、视频和 `shared_state` / `question` 执行。完整示例见 [Transformers 推理](https://valen-team.github.io/zh-Hans/docs/valen/quickstart/)。

首次加载会缓存原生模型权重；已有缓存后，在 `AutoModel.from_pretrained(...)` 中加入 `local_files_only=True` 可禁止下载，也可传入本地模型目录。这里使用原生 `Valen-*B` 权重，ONNX 模型通过下方 SDK 加载。

<a id="onnx-sdk"></a>

#### ONNX 与 SDK

使用 ONNX Runtime 在自己的应用中推理，无需安装 PyTorch 或 Transformers。Python 支持 CPU / NVIDIA CUDA，JavaScript / TypeScript 支持 Node.js CPU 与浏览器 WebGPU；C++ 当前提供 SystemOne 文本 FP32 CPU 集成。

本地 Python 推理需要 Python 3.11+；仅 HTTP 客户端支持 Python 3.10+。Python 首次加载会下载完整的所选变体，例如 0.8B FP32 约 3.505 GB；后续复用缓存，也可传入本地模型目录。

```bash
python -m pip install "valen-sdk[onnx]==0.3.0"
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

示例打印 0–1 的概率，值越大表示模型越认为客户在请求修改配送地址。

`q8-mlp` 可降低下载体积与文本内存，CUDA 使用 `cuda-fp16`，0.8B 浏览器使用专用 `web-portable`。

完整的图片/视频、SystemOne、CUDA、浏览器、本地 HTTP 服务与离线缓存示例见 [ONNX 与 SDK 指南](https://valen-team.github.io/zh-Hans/docs/valen/onnx/)。[Python SDK](https://github.com/Valen-Team/valen-sdk-python) · [JavaScript SDK](https://github.com/Valen-Team/valen-sdk-js) · [C++ 文本集成](https://github.com/Valen-Team/valen-sdk-cpp)。要在同一套问题上与 GPT、Claude 或 Gemini 对比，使用 [`Valen.llm`](https://valen-team.github.io/zh-Hans/docs/valen/onnx/#与大模型对比)。自己导出检查点见 [模块化 ONNX 导出](scripts/export/qwen/README-onnx-v2_zh.md)。

### 3. 评估 VisualDecisionBench

下载并解压媒体，然后运行[评估脚本](evaluation/visualdecisionbench/evaluate.py)：

```bash
hf download Valen-Team/VisualDecisionBench --repo-type dataset --local-dir data/VisualDecisionBench
python data/VisualDecisionBench/unpack_assets.py
python -m evaluation.visualdecisionbench.evaluate \
  --model Valen-Team/Valen-2B --data data/VisualDecisionBench \
  --output output/visualdecisionbench
```

添加 `--lora --revision lora-ckpt` 可评估旧 LoRA 模型。输出 `predictions.jsonl` 和 `metrics.json`，包含图像/视频、Choice/Noul/Score 分项与耗时。视频采样 16 帧；准确率仅统计硬标签题，软标签 Score 题参与概率指标计算。

### 4. 训练与微调

优先从完整 Valen-0.8B 使用 LoRA SFT；从 Qwen 基座开始的评分头 warmup / joint SFT 不是语言模型预训练。RLCD 为实验性路径。

[方法与配置](https://valen-team.github.io/zh-Hans/docs/valen/training/) · [完整可执行领域案例](https://valen-team.github.io/zh-Hans/cookbook/domain-finetuning/) · [保存、恢复和导出](https://valen-team.github.io/zh-Hans/docs/valen/checkpoints/)

<a id="community-projects"></a>

## 社区项目

以下公开项目接入了 Valen 模型或使用了 Valen 数据集。介绍依据链接中的资料整理；收录不代表合作、背书或已独立复现。使用前请查看各项目的许可证和支持的模型版本。

| 项目 | 与 Valen 的关系 | 证据与范围 |
| --- | --- | --- |
| [Jev-ToM](https://github.com/HKUST-KnowComp/Jev-ToM) | 研究评测 | 在心智理论基准上评测 Valen-Preview-0923，并在 [arXiv 预印本](https://arxiv.org/abs/2610.18610)中引用其模型卡；结果适用于该版本和论文所述评测设置。 |
| [vLLM Jev](https://github.com/mode-io/vllm-jev) | 模型部署 | 在其说明的 Linux 和 Mac 路径中，为 Valen-Preview-0923 提供文本、图片与短视频决策服务，详见[模型列表](https://github.com/mode-io/vllm-jev#supported-models)。 |
| [System1-Omni](https://github.com/InferMatrixLab/system1-omni) | 参考推理服务 | 提供 Valen-Preview-0923 的 [Python worker](https://github.com/InferMatrixLab/system1-omni/blob/main/recipe/valen/reference.md)，支持文本和单张图片；该路径不支持视频或原生 Valen CUDA 执行。GitHub 元数据未显示仓库许可证。 |
| [JevLight](https://github.com/usail-hkust/JevLight) | 仿真配置 | 交通信号仿真项目包含 [Valen 模型配置](https://github.com/usail-hkust/JevLight/tree/main/config/valen-2b)；这说明已配置接入，不代表实验结果已经发布。GitHub 元数据未显示仓库许可证。 |
| [reflex](https://github.com/ptlnextdoor/reflex) | 模型实验 | 在合成游戏画面上进行 [Valen-0.8B 参考与 MLX 实验](https://github.com/ptlnextdoor/reflex/blob/main/bench/infer/RESULTS.md)，作者也记录了局限。GitHub 元数据未显示仓库许可证。 |
| [LLM2Jev](https://github.com/Yinsongxu/LLM2Jev) | 数据集评测 | 使用 [Valen-Eval-Game](https://github.com/Yinsongxu/LLM2Jev#valen-eval-game) 对自己的推理流程进行推箱子单步评测。 |
| [tasksource](https://github.com/sileod/tasksource) | 数据集整理 | 将 Valen 游戏样本整理为类型化决策数据；划分和标签处理见其[数据审计](https://github.com/sileod/tasksource/blob/main/docs/jev/audits/game-expansion-audit.md)。 |

<a id="contributions"></a>

## 参与贡献

欢迎一起改进 Valen。你可以通过 [Issues](https://github.com/Valen-Team/Valen/issues) 反馈问题、分享应用场景和实验结果，也可以提交 [Pull Requests](https://github.com/Valen-Team/Valen/pulls) 改进代码与文档、补充训练数据或评测任务。积极为 Valen 作出贡献的开发者有机会加入核心开发团队。

欢迎扫描下方二维码加入 Valen 微信群，一起讨论项目、交流使用体验和实验结果。

<p align="center">
  <img src="assets/figures/wechat_1021.png" alt="Valen 微信讨论群二维码" width="200">
</p>

<a id="许可与致谢"></a>

## 许可与致谢

代码使用 [Apache 2.0](LICENSE) 许可。基础模型和来源数据集遵循各自的许可。

模型基于 [Qwen3.5](https://huggingface.co/Qwen/Qwen3.5-2B)，决策接口受 [TypeSafe Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev) 启发。

<a id="引用"></a>

## 引用

如果你在工作中使用 Valen，请引用本仓库：

```bibtex
@misc{valen,
  author       = {{Valen Team}},
  title        = {Valen},
  version      = {0.3.0},
  howpublished = {GitHub - Valen-Team/Valen: A multimodal System One decision model for text, images and video.},
  note         = {GitHub repository},
  year         = {2026},
  url          = {https://github.com/Valen-Team/Valen}
}
```

用户指南唯一维护在官网：[Documentation](https://valen-team.github.io/zh-Hans/docs/) · [Fine-tuning](https://valen-team.github.io/zh-Hans/docs/valen/training/) · [Probabilities](https://valen-team.github.io/zh-Hans/docs/valen/probabilities/) · [Cookbook](https://valen-team.github.io/zh-Hans/cookbook/).
