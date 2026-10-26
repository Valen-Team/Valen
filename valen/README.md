# 代码目录

命令入口位于包顶层，实际逻辑分为数据、模型、训练、评估四部分。下面的路径可直接用于定位实现；端到端流程见[架构说明](https://valen-team.github.io/zh-Hans/docs/valen/architecture/)。此处的 `valen/evaluation/` 负责通用逐题评估；仓库根目录另有[任务评测包](../evaluation/README.md)，实现 Sokoban 环境和完整游戏评测。

| 文件 | 主要对象或函数 | 职责 |
| --- | --- | --- |
| [data/schema.py](data/schema.py) | `candidates`、`target_distribution`、`read_jsonl` | 构造三类候选，校验标签，读取记录 |
| [data/types.py](data/types.py) | `QuestionSpec` | 两套架构共用的题目、候选和标签元数据 |
| [data/compilers/qwen.py](data/compilers/qwen.py) | `Compiler`、`CompiledState`、`Question`、`Branch` | Qwen 模板、媒体处理、分支和读取位置 |
| [data/compilers/dual_encoder.py](data/compilers/dual_encoder.py) | `ParallelCompiler`、`ParallelState` | 独立文本序列、单图预处理和视觉 mask |
| [modeling/factory.py](modeling/factory.py) | `build_model`、`build_compiler`、`get_backend` | 统一架构入口 |
| [modeling/interfaces.py](modeling/interfaces.py) | `ArchitectureBackend`、`Capabilities`、`Decision` | 编译、训练执行单元、推理和参考策略接口 |
| [modeling/qwen/](modeling/qwen/) | `ValenQwen`、`QwenBackend` | 模型、基座加载、LoRA、参数分组与冻结特征缓存 |
| [modeling/dual_encoder/](modeling/dual_encoder/) | `ValenDualEncoder`、`DualEncoderBackend` | 双编码器、融合、reader、基座适配与共享状态执行 |
| [configuration.py](configuration.py) | `flatten_config` | 分区配置转换为兼容现有 checkpoint 的运行配置 |
| [training/runner.py](training/runner.py) | `normalize_config`、`run` | 共用训练循环、打包预算、初始化与恢复、日志和保存 |
| [training/sft.py](training/sft.py) | `SFTObjective`、`question_loss` | 分布交叉熵、可选 Brier 与 Score RPS |
| [training/rlcd.py](training/rlcd.py) | `RLCDObjective`、`Rollout` | 候选采样、奖励与优势、裁剪损失、固定参考策略和特征复用 |
| [training/distributed.py](training/distributed.py) | `Distributed`、`epoch_shard`、`synchronize_gradients` | 按 rank 分片、训练参数广播、累积后的梯度求和 |
| [training/checkpoint.py](training/checkpoint.py) | `capture_rank_state`、`save_checkpoint`、`load_checkpoint` | 可训练参数和各 rank 的优化进度、随机状态 |
| [evaluation/inference.py](evaluation/inference.py) | `answer`、`predict` | logits 转概率、三类响应、推理 CLI |
| [evaluation/metrics.py](evaluation/metrics.py) | `question_metrics`、`summarize` | 逐题指标与各维度汇总 |
| [evaluation/evaluate.py](evaluation/evaluate.py) | `run` | 按记录分片评估、检查覆盖、写出预测与指标 |

## 调用关系

`train.py`、`inference.py`、`evaluate.py` 只调用对应子模块的 `main()`。公共调用通过 `modeling.factory` 选择架构；架构专用导入使用对应子包。`modeling.model`、`data.compiler` 和 `data.parallel_compiler` 保留旧导入入口。

训练中，`runner.run` 先构建模型、优化器和编译器，再把有标签记录编译成一个 pack。训练目标共用以下接口：

- `prepare(model, pack)`：SFT 返回空占位；RLCD 固定本批动作、旧策略概率、奖励、优势及参考概率。
- `loss_from_logits(logits, question, rollout)`：计算单题损失和日志项，不访问模型内部结构。
- `state_dict()`：SFT 返回 `None`；RLCD 返回需要随 checkpoint 保存的固定参考参数。

`num_iterations` 决定一个 pack 更新几次，`metric_names` 决定 runner 汇总哪些目标专用指标。新增训练目标时，除实现这些方法，还需在 `normalize_config` 和目标选择处注册。

架构适配器的 `training_units` 决定一次反传覆盖哪些题目。Qwen 每次返回一道题，反传后才计算下一题；双编码器一次返回整个 state 的题目，共享状态只构建一次。runner 按题目数和全局 state 数归一化后，对每个执行单元反传。适配器是普通 Python 对象，不给模型参数键增加前缀。

推理和评估都使用 `inference_units`。推理通过 `answer` 组装响应，评估额外计算 `question_metrics`。共享 state 的前向时间按题数摊分，并在报告中保留整个 state 的时间，不能把摊分值当作单题独立请求延迟。

## 测试对应关系

| 测试 | 主要覆盖 |
| --- | --- |
| [test_decisions.py](../tests/common/test_decisions.py) | 标签合法性、SFT/RPS、答案字段、决策头梯度 |
| [test_compiler.py](../tests/qwen/test_compiler.py) | UTF-8 读取、候选位置、Score 隔离、媒体展开、视频位置编码 |
| [test_checkpoint.py](../tests/common/test_checkpoint.py) | 参数保存、优化器与随机状态恢复 |
| [test_manifest.py](../tests/common/test_manifest.py) | 基础模型清单的优先级、旧文件读取与歧义检测 |
| [test_distributed.py](../tests/integration/test_distributed.py) | 无填充分片、不等数量 state、空闲 rank、梯度与恢复 |
| [test_sft_training.py](../tests/integration/test_sft_training.py) | 小模型 SFT 循环、旧配置处理与恢复 |
| [test_rlcd.py](../tests/common/test_rlcd.py) | 奖励、软标签、优势、裁剪梯度和参考 KL |
| [test_rlcd_training.py](../tests/integration/test_rlcd_training.py) | 三类任务、单/双进程连续训练与恢复的一致性 |
| [test_frozen_features.py](../tests/qwen/test_frozen_features.py) | RLCD 特征复用前后的损失、梯度和更新一致性 |
| [test_evaluate.py](../tests/common/test_evaluate.py) | 指标公式、有效样本分母、显式数据路径 |
| [test_model_naming.py](../tests/qwen/test_model_naming.py) | 模型标识与 checkpoint 参数键兼容 |

CPU 测试主要使用小模型替身。真实处理器测试需要本地 `models/Qwen3.5-2B/tokenizer.json` 等处理器文件；没有时跳过。真实权重加载、视觉梯度和 NCCL 恢复由 `scripts/smoke/` 中的 GPU 检查覆盖，命令见[脚本说明](../scripts/README.md)。

测试按 `tests/common/`、`tests/qwen/`、`tests/dual_encoder/`、`tests/integration/` 分组。架构接口测试还检查旧导入兼容、配置等价、逐题反传和共享状态的一次反传。

Qwen 模型类使用 `ValenQwen`，架构标识为 `qwen`；响应的 `model` 字段读取模型的 `model_name`，值为 `Valen`。Python 包、命令入口和下载模型清单使用 `valen` 命名。Qwen checkpoint 保持 `backbone.*` 与 `head.*` 参数键，双编码器也保留原有参数键；加载器还能读取模型目录中仅有一份、字段完整的旧清单文件。
