# Qwen decision heads / Qwen 决策头

`head_type` selects a complete candidate scorer. The generic source fallback `bilinear` preserves
existing checkpoint parameter names. Released complete models save Mixer/shared_state and must inherit their metadata. All heads return one FP32 logit per candidate.

| head_type | 输入 / Inputs | 默认结构 / Defaults |
| --- | --- | --- |
| `bilinear` | Decision + candidate endpoint | Two 256-dimensional projections and dot product |
| `mlp` | Decision + candidate endpoint | Separate LayerNorm, concatenate, 512 → 128 → 1 with GELU |
| `role_mlp` | Context, instruction, candidate description, Decision | Shared projection to 256, concatenate four roles, 1024 → 1 with GELU |
| `mixer` | Same four roles | Two blocks, token MLP 4 → 16 → 4, channel MLP 256 → 1024 → 256, candidate-role readout |

四角色头先对编译器标记的上下文、问题和候选描述区间求均值，第四个角色取 Decision 位置。
上下文包含公共 state 的文本和视觉位置；这些都是 Qwen 处理后的隐状态。平均池化可能丢失空间细节。
候选名称继续进入原有提示词，候选描述池化使用描述所覆盖的 token；边界 token 可能同时覆盖空格或分隔符。
历史 `question` 模式保留原有分词序列和 Score 的等级独立分支。
`shared_state` 模式使用一个共享序列，各题读取自身区间和 Decision；Score 的全部
等级在同一序列中一起评分。四种头的参数结构保持一致，详见
[共享 state 训练](https://valen-team.github.io/zh-Hans/docs/valen/qwen-shared-state/)。

Candidate rows share parameters and never mix inside the head. This does not make
the full Qwen model invariant to option order: causal backbone states already depend
on their prefix. Mixer token mixing operates over four semantic roles, not options.

`head_signature()` validates architecture dimensions and rejects incompatible stage
initializations. Backbone feature caching stores raw readouts before any trainable
normalization or projection, so RLCD policy/reference updates remain valid.

## Configuration / 配置

在现有 Qwen 训练配置的 `model` 段中设置 `head_type`，例如：

```json
{
  "head_type": "mixer",
  "head_width": 256,
  "head_layers": 2,
  "head_token_hidden_dim": 16,
  "head_channel_hidden_dim": 1024
}
```

This snippet is an architecture contract for head implementers, not a full training recipe. Saved complete-model settings take precedence over generic fallbacks; incompatible signatures are rejected. See [training methods](https://valen-team.github.io/docs/valen/training/) and [configuration](https://valen-team.github.io/docs/valen/configuration/).
