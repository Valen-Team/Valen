# Valen-Decision-1.5 architecture / Valen-Decision-1.5 架构

`Valen-Decision-1.5` is the generation that adds block-structured full attention and a grounding
head to the Qwen decision model. It is a set of configuration options on the existing `qwen`
architecture, not a separate backend: a 1.5 model is built, trained, exported and served through
the same code path as earlier checkpoints.

`Valen-Decision-1.5` 是在 Qwen 决策模型上新增分块全注意力与定位头的代际。它是现有 `qwen`
架构上的一组配置项，不是独立后端：1.5 模型与早期 checkpoint 走同一条构建、训练、导出与服务路径。

## What the generation label does and does not rename / 代际标签改与不改的部分

The loading contract is shared by every released checkpoint and does not carry a generation, so it
is unchanged. Renaming any of these would strand published checkpoints and the vLLM/SGLang adapters
that read them.

加载契约由所有已发布 checkpoint 共用，不随代际变化，因此保持原样。改动其中任何一项都会让
已发布的 checkpoint 以及读取它们的 vLLM / SGLang 适配器失效。

| Unchanged / 保持不变 | Value / 取值 |
| --- | --- |
| `model_type` | `valen_qwen` |
| Exported `auto_map` classes | `ValenQwenConfig`, `ValenQwenForDecisionMaking` |
| `architecture` config value | `qwen` |
| Attention backend name | `valen_block_bidirectional` |
| Backbone forward kwargs | `valen_prefix_lengths`, `valen_segments`, `valen_media_kwargs` |
| Export manifest `format` prefix | `valen_qwen_` |

`identity.py` adds the generation label the loading contract lacks:

`identity.py` 补上加载契约没有的代际标签：

```python
DECISION_MODEL_NAME   = "Valen-Decision-1.5"
DECISION_ARCH_SLUG    = "valen_decision_1_5"
DECISION_ARCH_VERSION = "1.5"
LEGACY_ARCH_VERSION   = "1.0"
```

`decision_arch_version(full_attention_mask, grounding)` returns `"1.5"` when either 1.5 feature is
in use and `"1.0"` otherwise. Both arguments default to the 1.0 values, so a caller that knows about
neither never reports 1.5. Exported configs gain `valen_decision_version`; bundles written before
that field existed derive it from the same two settings rather than assuming a default.

两个参数都以 1.0 的取值为默认，不知情的调用方不会误报 1.5。导出 config 新增
`valen_decision_version`；早于该字段的导出包按同样两项设置推导，而不是假定默认值。

## Block-structured full attention / 分块全注意力

`full_attention_mask` selects the mask. It applies **only to the full-attention layers**; the linear
attention layers stay causal. Block boundaries travel as backbone kwargs, so gradient checkpointing
recomputes with the same structure and nothing depends on global state.

`full_attention_mask` 选择掩码，**只作用于 full-attention 层**，线性注意力层保持因果。块边界通过
backbone kwargs 传入，梯度重算时结构一致，不依赖全局状态。

Each row has a shared prefix `[0, p)` — the compiler's `context_span`, i.e. the state — which is
bidirectional internally and never attends to the question region, so prefix KV stays shareable
across questions.

每行有一个共享前缀 `[0, p)`（编译器的 `context_span`，即 state），前缀内部双向且永不看到题目区，
因此前缀 KV 仍可在多题间共享。

| `full_attention_mask` | Question region / 题目区 |
| --- | --- |
| `causal` | Causal throughout / 全程因果 |
| `prefix_lm` | Causal, sees the full prefix / 因果，可见完整前缀 |
| `block_bidirectional` | One block, bidirectional inside, sees the full prefix / 整体一块，块内双向 |
| `question_blocks` | One block per question: prefix, shared glue and itself / 每题一块：前缀、公共胶水与本题 |
| `bbox_blocks` | Localization questions isolated; decision questions share one block / 定位题隔离，决策题共用一块 |

Segment ids come from the `shared_state` compiler: prefix `-1`, shared glue `0`, padding `-2`,
questions `1, 2, …`. `question_blocks` and `bbox_blocks` therefore require `qwen_execution=shared_state`.

段 id 由 `shared_state` 编译器给出：前缀 `-1`、公共胶水 `0`、padding `-2`、题目 `1, 2, …`。
因此 `question_blocks` 与 `bbox_blocks` 需要 `qwen_execution=shared_state`。

**One kernel for every shape.** Selecting a kernel by sequence length is not an option: the length
is each rank's own padded length, so ranks on opposite sides of a threshold would run different
kernels and deadlock in the next collective. The dense path handles all shapes.

**所有形状统一走一个 kernel。** 不能按序列长度选 kernel：长度是各 rank 自己的 padding 长度，
阈值两侧的 rank 会选到不同 kernel，并在下一次集合通信里死锁。dense 路径处理所有形状。

## Grounding head / 定位头

`grounding_head` enables the `bbox` question type. Localization is treated as a decision over image
tokens: the question's decision readout scores every visual token of the referenced image (softmax
over tokens), and every token regresses the distances from its cell centre to the four box edges
(FCOS-style). The predicted box is the regression of the most probable token; the confidence is the
probability mass of tokens whose centres fall inside that box. Coordinates are normalized to `[0, 1]`
against the original image size.

定位即在图像 token 上做决策：读出位为被引用图像的每个视觉 token 打分（token 维 softmax），
每个 token 回归其格点中心到框四边的距离（FCOS 风格）。预测框取概率最大 token 的回归结果，
置信度为中心落在预测框内的 token 概率质量。坐标按原图宽高归一化到 `[0, 1]`。

Loss: `CE + 5·L1 + 2·GIoU`.

## Configuration / 配置

```json
{
  "architecture": "qwen",
  "qwen_execution": "shared_state",
  "full_attention_mask": "bbox_blocks",
  "grounding_head": {"width": 512, "hidden": 1024}
}
```

Omitting `full_attention_mask` and `grounding_head` reproduces the previous generation, which is why
both default to the 1.0 values throughout. When `model_path` is a published Valen model, both are
inherited from its `config.json` (`valen_full_attention_mask`, `valen_grounding`) together with the
trained grounding head, so fine-tuning a Valen-Decision-1.5 model keeps its architecture; a value set
in the training config takes precedence. This snippet is an architecture contract, not a training
recipe; see [training methods](https://valen-team.github.io/docs/valen/training/) and
[configuration](https://valen-team.github.io/docs/valen/configuration/), and
[HEADS.md](HEADS.md) for the decision heads.

省略 `full_attention_mask` 与 `grounding_head` 即复现上一代，这也是两者在各处都以 1.0 取值为默认的原因。
`model_path` 指向已发布的 Valen 模型时，两者连同已训练的定位头一起从其 `config.json`
（`valen_full_attention_mask`、`valen_grounding`）继承，微调 Valen-Decision-1.5 模型即沿用其架构；训练配置里显式设置的值优先。
本段是架构契约而非训练配方，训练方法与配置见上述链接，决策头见 [HEADS.md](HEADS.md)。
