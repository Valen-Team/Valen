"""Generation identity for the Valen decision architecture. / Valen 决策架构的代际标识。

The loading contract is fixed and shared by every released checkpoint: ``model_type`` stays
``valen_qwen``, the exported ``auto_map`` keeps pointing at ``ValenQwenConfig`` and
``ValenQwenForDecisionMaking``, and the attention backend keeps its registered name. Those strings
live in published ``config.json`` files, so renaming any of them would strand existing checkpoints
and the vLLM/SGLang adapters that read them.

What this module adds is the *generation* label, which the loading contract does not carry. A
bundle's generation was previously only implicit in whether ``valen_full_attention_mask`` and
``valen_grounding`` were present; ``decision_arch_version()`` makes it explicit so exports,
manifests and serving code can name it.

加载契约是所有已发布 checkpoint 共用的，不随代际变化：``model_type`` 仍为 ``valen_qwen``，
导出的 ``auto_map`` 仍指向 ``ValenQwenConfig`` 与 ``ValenQwenForDecisionMaking``，注意力后端
也保留已注册的名字。这些字符串写在已发布的 ``config.json`` 里，改名会让旧 checkpoint 和读取
它们的 vLLM / SGLang 适配器失效。

本模块只补上加载契约没有的「代际」标签。此前只能从 ``valen_full_attention_mask`` 和
``valen_grounding`` 是否存在来间接判断代际，``decision_arch_version()`` 把它显式化。
"""

# Display name, machine slug and version of the current decision architecture.
# 当前决策架构的展示名、机器可读 slug 与版本号。
DECISION_MODEL_NAME = "Valen-Decision-1.5"
DECISION_ARCH_SLUG = "valen_decision_1_5"
DECISION_ARCH_VERSION = "1.5"

# The generation a checkpoint predates this label with. / 早于本标签的代际。
LEGACY_ARCH_VERSION = "1.0"

# Features that 1.5 introduced over 1.0. A checkpoint using any of them is 1.5.
# 1.5 相对 1.0 新增的能力，命中任意一项即为 1.5。
DECISION_1_5_FEATURES = ("block_attention_mask", "grounding_head")


def decision_arch_version(full_attention_mask=None, grounding=None):
    """Generation of a model with these settings. / 给定设置对应的代际。

    ``full_attention_mask`` is the exported ``valen_full_attention_mask`` (``"causal"`` for 1.0),
    ``grounding`` the exported ``valen_grounding`` (``None`` when the head is absent). Both default
    to the 1.0 values so callers that know nothing about either keep reporting 1.0.
    """
    if (full_attention_mask or "causal") != "causal" or grounding:
        return DECISION_ARCH_VERSION
    return LEGACY_ARCH_VERSION


def decision_arch_features(full_attention_mask=None, grounding=None):
    """Which 1.5 features these settings use, in ``DECISION_1_5_FEATURES`` order. / 命中的 1.5 能力。"""
    used = []
    if (full_attention_mask or "causal") != "causal":
        used.append("block_attention_mask")
    if grounding:
        used.append("grounding_head")
    return tuple(used)


def decision_identity(full_attention_mask=None, grounding=None):
    """Identity block for export provenance and manifests. / 导出溯源与 manifest 用的标识块。"""
    version = decision_arch_version(full_attention_mask, grounding)
    return {"decision_model": DECISION_MODEL_NAME if version == DECISION_ARCH_VERSION else "Valen-Decision-1.0",
            "decision_arch_version": version,
            "decision_arch_features": list(decision_arch_features(full_attention_mask, grounding))}
