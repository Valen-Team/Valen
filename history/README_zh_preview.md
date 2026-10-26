<p align="center">
  <img src="assets/branding/valen-logo-v5-compact.svg" alt="Valen — 青绿色光圈、前进箭头与大写字标" width="720"><br>
  <img src="assets/branding/valen-slogan.svg" alt="System One Model, now with vision." width="700">
</p>

<p align="center">
  A multimodal decision model inspired by <a href="https://typesafe.ai/blog/introducing-system-one-models-and-jev">Jev</a> — text, images and video in; decision probabilities out.
</p>

<p align="center">
  <a href="https://huggingface.co/Valen-Team"><img src="https://img.shields.io/badge/Hugging_Face-Valen_Team-FFD21E?style=flat&amp;logo=huggingface&amp;logoColor=FFD21E&amp;labelColor=555555" alt="Hugging Face 组织：Valen Team"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-80A51B?style=flat&amp;labelColor=555555" alt="许可证：Apache 2.0"></a>
  <a href="pyproject.toml"><img src="https://img.shields.io/badge/Python-3.10%2B-1683BB?style=flat&amp;logo=python&amp;logoColor=FFD43B&amp;labelColor=555555" alt="Python 3.10+"></a>
  <a href="configs/train/"><img src="https://img.shields.io/badge/Training-SFT_%7C_RLCD-8A63B8?style=flat&amp;labelColor=555555" alt="SFT 与实验性 RLCD"></a>
  <br>
  <a href="https://huggingface.co/Valen-Team/Valen-Preview-0923"><img src="https://img.shields.io/badge/Model-Preview_0923-E88B23?style=flat&amp;logo=huggingface&amp;logoColor=FFD21E&amp;labelColor=555555" alt="模型：Valen-Preview-0923"></a>
  <a href="https://huggingface.co/spaces/yuhangzang/Valen-Preview-0923"><img src="https://img.shields.io/badge/Space-Try_Demo-009C83?style=flat&amp;logo=huggingface&amp;logoColor=FFD21E&amp;labelColor=555555" alt="在线体验 Valen：Hugging Face Spaces"></a>
  <a href="https://huggingface.co/datasets/Valen-Team/Valen-Training-General-100k"><img src="https://img.shields.io/badge/Train-General_100k-2185B5?style=flat&amp;logo=huggingface&amp;logoColor=FFD21E&amp;labelColor=555555" alt="训练集：Valen-Training-General-100k"></a>
  <a href="https://huggingface.co/datasets/Valen-Team/Valen-Eval-General-5k"><img src="https://img.shields.io/badge/Eval-General_5k-2185B5?style=flat&amp;logo=huggingface&amp;logoColor=FFD21E&amp;labelColor=555555" alt="评测集：Valen-Eval-General-5k"></a>
  <a href="https://huggingface.co/datasets/Valen-Team/Valen-Eval-Game"><img src="https://img.shields.io/badge/Eval-Game-2185B5?style=flat&amp;logo=huggingface&amp;logoColor=FFD21E&amp;labelColor=555555" alt="游戏评测集：Valen-Eval-Game"></a>
</p>

<p align="center">
  <a href="README.md">English</a> · <b>简体中文</b><br>
  <a href="#简介">简介</a> · <a href="#演示">演示</a> · <a href="#模型下载">模型下载</a> · <a href="#实验结果">评测结果</a> · <a href="#快速开始">快速开始</a>
</p>

<a id="简介"></a>

## ✨ 简介

Valen（万澜）将视觉感知引入 System One 决策。受 [Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev) 启发，它根据任务指令，对文本、图像和视频中的信息进行判断，直接输出给定候选的概率分布，为程序提供结构化的决策接口。模型以 Qwen3.5-0.8B/2B 为骨干，通过共享决策头完成评分，无需生成答案 token。仓库提供模型实现、数据处理、SFT 与实验性 RLCD 训练，以及推理和评估命令，支持使用自有数据训练。

<a id="演示"></a>

## 🎬 演示

在 [Hugging Face Space](https://huggingface.co/spaces/yuhangzang/Valen-Preview-0923) 在线体验 **Valen-Preview-0923**。

<a id="demo-comparison"></a>

### 与 27B 生成模型对比

**Valen-Preview-0923** 读取棋盘图像，在每一步选择移动方向。在相同关卡，Valen-Preview-0923 用 9 次决策通关，累计决策耗时为 1.13 秒。Qwen3.8-27B-FP8 的 thinking 模式耗时 198.05 秒，no-thinking 模式未通关。

<p align="center">
  <img src="assets/demos/sokoban-model-comparison.gif" alt="Valen-Preview-0923 与 Qwen3.8-27B-FP8 的 no-thinking 和 thinking 模式并排对比。" width="1000"><br>
</p>

### 图像模糊与动作置信度

高斯模糊展示 **Valen-Preview-0923** 如何应对视觉细节的减少，决策与置信度动态变化的情况。清晰图像上的决策置信度为 91.6%，最强模糊下为 19.2%。

<p align="center">
  <img src="assets/demos/valen-preview-0923-blur-confidence.gif" alt="Valen-Preview-0923 在九档高斯图像模糊下的动作概率和决策置信度。" width="1000"><br>
</p>

### 四局并行能力展示

四条成功的 **Valen-Preview-0923** 轨迹以 1× 速度并排播放，不做加速。每局需要 7–10 次决策，平均每步 122–128 毫秒，四局均于 1.24 秒内完成。

<p align="center">
  <img src="assets/demos/sokoban-four-game-showcase.gif" alt="四局 Valen-Preview-0923 成功轨迹以记录速度并行播放。" width="1000"><br>
</p>

<a id="模型下载"></a>

## 📥 模型下载

运行 Valen 需要同时下载 **Valen checkpoint** 和 **Qwen3.5-2B Base 模型**。

| 模型 | 用途 | 下载 |
| --- | --- | --- |
| Valen-Preview-0923 | Valen checkpoint | [🤗 Hugging Face](https://huggingface.co/Valen-Team/Valen-Preview-0923) |

数据集：[General 100k 训练集](https://huggingface.co/datasets/Valen-Team/Valen-Training-General-100k) · [General 5k 评测集](https://huggingface.co/datasets/Valen-Team/Valen-Eval-General-5k) · [Sokoban 训练与评测集](https://huggingface.co/datasets/Valen-Team/Valen-Eval-Game)。

<a id="实验结果"></a>

## 📊 评测结果

**更低的延迟，更高的准确率。** 四个子图从左到右依次比较 General 准确率、General 耗时、推箱子准确率和推箱子耗时，每图展示 Qwen3.5-0.8B、Qwen3.5-2B 与一个 Valen 2B RL checkpoint。General 使用 5,000 道题，来自多个 VQA 数据集；推箱子使用来自 100 关的 500 道单步题。

<p align="center">
  <a href="assets/figures/evaluation-results.png"><img src="assets/figures/evaluation-results.png" alt="四个子图比较 Qwen3.5-0.8B、Qwen3.5-2B 和 Valen 的 General 与推箱子准确率及平均端到端耗时。General 使用通用 RL checkpoint，推箱子使用 Valen-Preview-0923。" width="1400"></a>
</p>

训练数据、Model Card 和 loss 曲线见[技术说明](https://valen-team.github.io/cookbook/validation/archive/historical-experiments-zh-Hans.md)。

<a id="快速开始"></a>

## 🚀 快速开始

先按[环境要求](https://github.com/Valen-Team/Valen/blob/453f221/docs/technical_zh.md)安装依赖，再下载 Preview checkpoint 及其对应的 Qwen3.5-2B Base 模型。

```bash
# 下载 Valen-Preview-0923 checkpoint。
hf download Valen-Team/Valen-Preview-0923 --local-dir models/Valen-Preview-0923

# 下载 Qwen3.5-2B Base 模型。
hf download Qwen/Qwen3.5-2B --local-dir models/Qwen3.5-2B

# 使用自己的配置训练模型。
python -m valen.train \
  --config configs/train/qwen/sft_warmup.json

# 加载下载好的 Valen-Preview-0923 进行推理。
python -m valen.inference \
  --checkpoint models/Valen-Preview-0923 \
  --data data/smoke/train.jsonl \
  --output output/sft_warmup/predictions.jsonl

# 用同一组合成样本检查评估流程。
python -m valen.evaluate \
  --checkpoint models/Valen-Preview-0923 \
  --data data/smoke/train.jsonl \
  --output output/sft_warmup/smoke_eval
```

`data/smoke` 有少量简单题目，仅仅用于测试功能是否正常运行。

<details>
<summary>一条带图片输入的标注记录</summary>

JSONL 每行是一条记录。下例使用仓库里的[评测总览图](assets/figures/evaluation-results.png)，假设保存为仓库根目录的 `example.jsonl`。

```json
{
  "group_id": "evaluation-general-2b",
  "request": {
    "state": {
      "messages": [{
        "role": "user",
        "content": [
          {"type": "text", "text": "比较图中 General 任务上 2B 模型的准确率和平均单题耗时。"},
          {"type": "image_url", "image_url": {"url": "assets/figures/evaluation-results.png"}}
        ]
      }]
    },
    "questions": {
      "best_2b": {
        "type": "choice",
        "instructions": "在 General 任务上，平均单题耗时低于 200 ms 的 2B 模型中，哪个准确率最高？",
        "criteria": {
          "qwen": "Qwen3.5-2B",
          "valen": "Valen-Preview-0923"
        }
      }
    }
  },
  "targets": {
    "best_2b": {"probabilities": {"qwen": 0.0, "valen": 1.0}}
  }
}
```
</details>

<a id="参与贡献"></a>

## 🤝 参与贡献

欢迎一起改进 Valen。你可以通过 [Issues](https://github.com/Liuziyu77/Valen/issues) 反馈问题、分享应用场景和实验结果，也可以提交 [Pull Requests](https://github.com/Liuziyu77/Valen/pulls) 改进代码与文档、补充训练数据或评测任务。

欢迎扫描下方二维码加入 Valen 微信群，一起讨论项目、交流使用体验和实验结果。

<p align="center">
  <img src="assets/figures/wechat_1013.jpg" alt="Valen 微信讨论群二维码" width="200">
</p>

<a id="许可与致谢"></a>

## 📄 许可与致谢

代码使用 [Apache 2.0](LICENSE) 许可。基础模型和来源数据集遵循各自的许可。

模型基于 [Qwen3.5](https://huggingface.co/Qwen/Qwen3.5-2B)，决策接口受 [TypeSafe Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev) 启发。
