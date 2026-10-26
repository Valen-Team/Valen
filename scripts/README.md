# Valen 脚本

所有命令在仓库根目录执行，`python` 指已安装项目依赖的 Python 3.10+ 环境。

| 目录 | 工具 |
| --- | --- |
| `setup/` | 安装依赖，下载并核验基座模型 |
| `train/` | 按配置运行 SFT/RLCD，本机启动与集群提交 |
| `data/` | 生成合成样例、筛选双编码器训练检查子集 |
| `eval/` | 从通用实验快照绘制额外的详细图 |
| `export/qwen/` | 合并完整权重、模块化 ONNX 导出与验证 |
| `smoke/qwen/`、`smoke/dual_encoder/` | 各架构的训练与恢复集成检查 |

<a id="installation"></a>

## 安装

```bash
bash scripts/setup/bootstrap.sh
source .venv/bin/activate
```

已有 Conda 或 virtualenv 环境时，激活后手动安装：

```bash
python -m pip install --upgrade pip
python -m pip install --index-url https://download.pytorch.org/whl/cu124 \
  torch==2.6.0 torchvision==0.21.0
python -m pip install -e '.[test]'
```

训练配置使用 BF16，需要支持 BF16 的 NVIDIA GPU。`bootstrap.sh` 默认用 `python3` 创建 `.venv`，可用 `VJ_PYTHON` 指定解释器。

## 模型与数据

```bash
python scripts/setup/prepare_model.py
```

下载脚本固定使用 `Qwen/Qwen3.5-2B`，默认 revision 为 `15852e8c16360a2fea060d615a32b45270f8a8fc`。它下载快照，逐个检查 `.safetensors` 权重与 Hub 上的大小、摘要，加载本地 config/processor，然后生成 `valen_manifest.json`，记录 revision、依赖版本、权重摘要和媒体默认配置。

| 参数 | 默认值 / 用途 |
| --- | --- |
| `--output` | 默认 `models/Qwen3.5-2B`；自定义后需同步修改训练配置 |
| `--revision` | 指定要解析并下载的 revision |
| `--verify-local` | 核验输出目录内已有文件并重写清单，不下载权重；仍需访问 Hub 获取元数据 |

脚本固定下载 2B，与主 README 和八份训练示例配置一致。使用其他规格时，需另行下载完整快照，并修改训练配置中的 `model_path`。下载完成后，训练使用本地文件加载，不会补齐缺失权重。

`data/smoke/` 已包含合成样例；重新生成可运行 `python scripts/data/make_smoke_data.py`，它会覆盖这些样例。自有数据格式见[数据说明](https://valen-team.github.io/zh-Hans/docs/valen/data-format/)。

## 训练与推理

```bash
python -m valen.train --config configs/train/qwen/sft_warmup.json

VALEN_GPUS=4 bash scripts/train/launch.sh configs/train/qwen/rlcd_joint.json \
  --initialize output/sft_warmup/latest

python -m valen.inference \
  --checkpoint output/rlcd_joint/latest \
  --data data/smoke/train.jsonl \
  --output output/rlcd_joint/predictions.jsonl
```

`launch.sh` 支持两种架构和两种训练目标；`VALEN_GPUS` 指定本机进程数（默认 1），`VALEN_PYTHON` 指定解释器（默认仓库 `.venv/bin/python`），`CUDA_VISIBLE_DEVICES` 选择 GPU。也接受原有的 `VJ_GPUS`、`VJ_PYTHON`。不传配置时使用 Qwen 的 `sft_joint.json`；配置参数之后的参数原样传给训练 CLI。`tokens_per_step` 是每卡预算。推理输出文件的父目录需已存在。旧入口 `launch_sft.sh` 转发到该脚本，并保留默认 2 进程。

集群提交需要可用的 `rjob` 客户端和本机配置。复制 `scripts/train/cluster.env.example` 到仓库根目录的 `.env.cluster`，填写 namespace、配额组、镜像和挂载；该本机文件已被 Git 忽略。提交脚本自动加载它，也支持直接通过环境变量配置。

`RJOB_BIN` 默认从 PATH 查找 `rjob`。`RJOB_NAMESPACE`、`RJOB_CHARGED_GROUP`、`RJOB_IMAGE` 为必填；`RJOB_MOUNTS` 每行一个挂载，路径含空格时仍作为一个参数传递。仓库、模型和数据须在提交节点及工作节点的相同路径可见。

`submit.sh <config> [训练参数]` 默认申请 1 张 GPU，可用 `VALEN_GPUS` 修改；`VALEN_TIMEOUT` 默认 1800 秒。`VALEN_PYTHON` 默认使用仓库 `.venv/bin/python`。日志和退出码写入 `artifacts/jobs/<job>/`。`submit_experiment.sh` 默认申请 8 卡，使用 `VALEN_RUN_NAME` 命名，输出到 `VALEN_EXPERIMENT_ROOT`（默认 `next-generation-exp`）。

```bash
bash scripts/train/submit.sh configs/train/qwen/sft_warmup.json
bash scripts/train/submit.sh smoke qwen --output artifacts/qwen_smoke_run
bash scripts/train/submit.sh smoke dual_encoder --output artifacts/dual_smoke_run
```

两种 smoke 都可以用 `--config` 和 `--output` 指定输入配置与独立输出目录。旧的 `submit_dual.sh`、`dual_entry.sh` 和 smoke 文件路径保留转发入口。

八份基础配置覆盖 SFT/RLCD × 四种 stage，均使用合成数据和短程设置。正式训练需调整 `data`、`model_path`、`epochs`、`max_steps` 和 `output`。每次独立训练使用新的输出目录；初始化与恢复见[训练指南](../valen/training/README.md)。

## ONNX 导出

直接使用发布模型见 [ONNX 与 SDK](https://valen-team.github.io/zh-Hans/docs/valen/onnx/)，无需安装本仓库的训练或导出依赖。

导出自己的完整 Valen 检查点时，使用模块化导出器，在独立环境中准备依赖：

```bash
python -m venv export-env
. export-env/bin/activate
python -m pip install -r scripts/export/qwen/requirements-onnx-v2-export.txt
git clone https://github.com/microsoft/onnxruntime-genai.git /path/to/onnxruntime-genai
git -C /path/to/onnxruntime-genai checkout 6b7ff3ca3efa15da219050fdb4ecf8bed67875af
python scripts/export/qwen/onnx_v2.py \
  --source /path/to/Valen-0.8B --output /path/to/new-cpu-bundle \
  --upstream /path/to/onnxruntime-genai
```

从 Valen 仓库根目录运行，输出目录须为空或不存在。CPU 默认导出 FP32；CUDA 使用 `--provider cuda --precision fp16`。使用 `onnx_v2_quantize.py` 添加仅量化 MLP 的 `q8-mlp` 候选；0.8B 浏览器图由 `onnx_v2_web.py` 单独生成。

产物包含 `valen_onnx.json`、embedding、vision、融合决策图及外部权重，使用独立的 Python / JavaScript SDK 加载。导出完成后仍需完成原生输出对齐、质量与目标设备验证。已发布变体为 `fp32`、`q8-mlp`、`cuda-fp16`，以及 0.8B 的 `web-portable`。

环境、完整参数与模板来源见 [模块化 ONNX 导出](export/qwen/README-onnx-v2_zh.md)（[English](export/qwen/README-onnx-v2.md)）。发布的 HF 模型仓库保存资产、清单和使用/验证文档；SDK 源码分别维护在 [Python](https://github.com/Valen-Team/valen-sdk-python)、[JavaScript](https://github.com/Valen-Team/valen-sdk-js)、[C++](https://github.com/Valen-Team/valen-sdk-cpp) 仓库。

## 命令参数

| 命令 | 必填 | 可选 |
| --- | --- | --- |
| `python -m valen.train` | `--config` | `--method {sft,rlcd}`、`--output`、互斥的 `--initialize` / `--resume` |
| `python -m valen.inference` | `--checkpoint`、`--data`、`--output`（文件） | `--device`（默认 `cuda`）、`--calibration` |
| `python -m valen.evaluate` | `--checkpoint`、`--data`、`--output`（目录） | `--device`（默认 `cuda`） |

训练设备、学习率、数据路径等在 JSON 中设置，不能直接使用 `train --device` 或 `train --data`。详细字段见[配置参考](https://valen-team.github.io/zh-Hans/docs/valen/configuration/)。各入口可通过 `--help` 查看参数，需要先安装 Python 依赖。

## 通用评估

```bash
# 合成样例只检查流程；正式评估替换为独立测试集。
python -m valen.evaluate \
  --checkpoint output/sft_warmup/latest \
  --data data/smoke/train.jsonl \
  --output output/sft_warmup/smoke_eval

# 本机四卡评估，同一有标签问题只处理一次。
python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=4 \
  -m valen.evaluate \
  --checkpoint output/sft_warmup/latest \
  --data data/smoke/train.jsonl \
  --output output/sft_warmup/smoke_eval_4gpu
```

输出包括逐题预测 `predictions.jsonl`、每卡分片及汇总 `metrics.json`。指标包含硬标签准确率、NLL、Brier，以及适用的二分类 F1、Score 误差和 RPS；按任务、模态、语言及领域汇总。命令始终使用显式 `--data` 参数指定的数据。评估固定使用温度 1，没有 `--calibration` 参数；指标公式和时间统计的范围见[推理与评估](https://valen-team.github.io/zh-Hans/docs/valen/evaluation/)。

## 实验图

`scripts/eval/plot_general.py` 读取 `assets/figures/general/results.json`，在指定目录输出四张详细图的 PNG、SVG 和 PDF。需要另行安装 Matplotlib，不需要模型、GPU 或原始评测目录。

```bash
python -m pip install matplotlib
python scripts/eval/plot_general.py --output-dir /tmp/valen-general-details
```

`--output-dir` 是必填参数，避免重绘结果混入仓库中的展示图。数据来源与计时口径见[结果快照](../assets/figures/general/results.json)。

## 验证

```bash
python -m pytest -q

# 以下检查需要已下载 2B 模型和相应数量的 GPU。
python scripts/smoke/gpu_smoke.py
python -m torch.distributed.run --standalone --nnodes=1 \
  --nproc_per_node=2 --max_restarts=0 scripts/smoke/multi_gpu_smoke.py
python -m torch.distributed.run --standalone --nnodes=1 \
  --nproc_per_node=2 --max_restarts=0 scripts/smoke/rlcd_gpu_smoke.py
```

CPU 测试不需要模型权重；与真实处理器相关的测试在没有本地处理器文件时会跳过。GPU 检查实际执行短程训练并验证恢复结果，产物写入 `artifacts/` 或 `output/`，不衡量模型质量。

| 脚本 | GPU 数 | 主要检查 | 结果文件 |
| --- | --- | --- | --- |
| `gpu_smoke.py` | 1 | 基础 embedding 加载、三类输入/输出、参数组梯度、保存重载、stage 初始化 | `artifacts/gpu_smoke/result.json` |
| `multi_gpu_smoke.py` | 必须为 2 | SFT 梯度同步、不等分片、空闲 rank、恢复与连续训练一致 | `artifacts/multi_gpu_smoke/result.json` |
| `rlcd_gpu_smoke.py` | 必须为 2 | RLCD 更新、固定参考策略、三类决策、恢复与连续训练一致 | `output/rlcd_gpu_smoke/result.json` |

这些脚本使用固定的模型路径和输出目录，重新运行会覆盖对应产物，不接受训练 CLI 的参数。CPU 测试的文件与覆盖范围见[代码目录](../valen/README.md#测试对应关系)。

## 常见问题

| 报错或现象 | 检查与处理 |
| --- | --- |
| 找不到模型、处理器，或 `Incomplete pretrained model load` | 确认下载目录完整、`model_path` 指向该目录、依赖与 `pyproject.toml` 一致；加载器会拒绝缺失或不匹配的权重 |
| `Media hash mismatch` / 媒体文件不存在 | 路径按 JSONL 所在目录解析；确认文件内容与 `assets` 记录对应，替换素材后同步更新清单 |
| `tokens exceed max_length` | 某个分支过长；缩短 state/候选描述，或调整媒体处理参数与长度上限 |
| `State ... exceeds tokens_per_step` | 完整 state 的所有分支超过每卡预算；适当增大预算或减小单条记录的分支计算量，程序不会拆开 state |
| CUDA 显存不足 | 每卡加载完整模型，增加 GPU 数不拆分模型；检查 stage、媒体长度和 RLCD 参考模型。RLCD warmup 可用 `cache_frozen_features=true` 复用特征 |
| `Text stage requires a text-only dataset` | `text` stage 收到了有标签的媒体题；使用纯文本数据，或为多模态训练选择 `joint` |
| `Initialization would freeze previously changed parameters` | 新 stage 没有包含原 checkpoint 的全部可训练参数；例如 joint 不能初始化到 warmup |
| `Resume config/data changed` / world size 不匹配 | 保持原配置、数据和进程数；更换实验条件使用 `--initialize` 和新输出目录 |
| 恢复后没有新增 step | 已到 `epochs` 或 `max_steps` 上限；在配置副本中提高已用尽的上限 |
| `No optimizer step` | 数据没有有效标签，或 `epochs` / `max_steps` 为 0 |
| 图片和视频指标都归到 `media` | 在记录中填写 `meta.modality`；缺省时评估不区分图片与视频 |

## Caption 数据准备

安装可选依赖 `python -m pip install -e '.[caption-data]'`。源索引、输出目录和凭据文件通过参数配置：

```bash
python scripts/data/prepare_caption.py \
  --source-root /datasets/caption-indices \
  --caption-root data \
  --credentials-config "$HOME/petreloss.conf"
```

也可设置 `VALEN_CAPTION_SOURCE_ROOT`、`VALEN_CAPTION_ROOT` 和 `VALEN_CREDENTIALS_CONFIG`。默认输出到 `data/train_caption/` 与 `data/eval_caption/`；自定义目录后，需要同步修改自己的预训练配置副本。凭据配置中的 `langchao` / `langchao2` 段须对应索引 URI 的存储集群，包含 `host_base`、`access_key` 和 `secret_key`；凭据文件放在仓库外。

`sft_with_caption_preparation.py` 使用同样的目录参数。监控注册项可保存 `caption_root`，恢复时会沿用它；否则读取 `VALEN_CAPTION_ROOT`，默认 `data`。

## Qwen 决策头

四种评分头的输入、配置和阶段初始化说明见 [Qwen 决策头](../valen/modeling/qwen/HEADS.md)。统一通过 `python -m valen.train --config ...` 训练，通过 `python -m valen.evaluate` 评估。

Python 监控脚本读取环境变量 `RJOB_BIN`。使用本机配置时，先运行 `source .env.cluster` 再启动监控。集群脚本不是通用调度器适配层；使用 Slurm 等环境时，可直接运行本机 `launch.sh` 或自行包装提交命令。
