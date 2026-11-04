"""Generation identity of the decision architecture. / 决策架构的代际标识。"""
from pathlib import Path

import pytest

from valen.modeling.qwen.identity import (
    DECISION_ARCH_SLUG,
    DECISION_ARCH_VERSION,
    DECISION_MODEL_NAME,
    LEGACY_ARCH_VERSION,
    decision_arch_features,
    decision_arch_version,
    decision_identity,
)


def test_names_are_stable():
    assert DECISION_MODEL_NAME == "Valen-Decision-1.5"
    assert DECISION_ARCH_SLUG == "valen_decision_1_5"
    assert DECISION_ARCH_VERSION == "1.5"
    assert LEGACY_ARCH_VERSION == "1.0"


def test_defaults_are_the_previous_generation():
    """A caller that knows about neither setting must not claim 1.5. / 不知情的调用方不应报 1.5。"""
    assert decision_arch_version() == LEGACY_ARCH_VERSION
    assert decision_arch_features() == ()


@pytest.mark.parametrize("mask", ["block_bidirectional", "prefix_lm", "question_blocks", "bbox_blocks"])
def test_block_masks_are_the_new_generation(mask):
    assert decision_arch_version(mask) == DECISION_ARCH_VERSION
    assert decision_arch_features(mask) == ("block_attention_mask",)


def test_grounding_head_alone_is_the_new_generation():
    assert decision_arch_version("causal", {"width": 512, "hidden": 1024}) == DECISION_ARCH_VERSION
    assert decision_arch_features("causal", {"width": 512}) == ("grounding_head",)


def test_causal_without_grounding_stays_legacy():
    assert decision_arch_version("causal", None) == LEGACY_ARCH_VERSION
    assert decision_arch_version(None, None) == LEGACY_ARCH_VERSION


def test_both_features_are_reported_in_order():
    assert decision_arch_features("bbox_blocks", {"width": 512}) == ("block_attention_mask", "grounding_head")


def test_identity_block_shape():
    new = decision_identity("bbox_blocks", {"width": 512})
    assert new == {"decision_model": "Valen-Decision-1.5", "decision_arch_version": "1.5",
                   "decision_arch_features": ["block_attention_mask", "grounding_head"]}
    legacy = decision_identity()
    assert legacy["decision_model"] == "Valen-Decision-1.0"
    assert legacy["decision_arch_version"] == "1.0"
    assert legacy["decision_arch_features"] == []


def test_exported_bundle_derives_the_version_when_absent(tmp_path, monkeypatch):
    """Bundles exported before the field existed must still report their generation.

    早于该字段的导出包仍应报出正确代际。导出模板依赖打包时才复制进去的同级模块，
    所以这里用真实的 bundle_runtime 产物来测。
    """
    pytest.importorskip("transformers")
    import sys
    from valen.modeling.qwen.hf_export import bundle_runtime

    root = Path(__file__).resolve().parents[2]
    bundle_runtime(root, tmp_path / "valen_runtime")
    monkeypatch.syspath_prepend(str(tmp_path))
    for name in [key for key in sys.modules if key.startswith("valen_runtime")]:
        del sys.modules[name]
    from valen_runtime.configuration_valen import ValenQwenConfig

    legacy = ValenQwenConfig(qwen_execution="question")
    assert legacy.valen_decision_version == "1.0"

    inferred = ValenQwenConfig(valen_full_attention_mask="bbox_blocks")
    assert inferred.valen_decision_version == "1.5"

    grounded = ValenQwenConfig(valen_grounding={"width": 512, "hidden": 1024})
    assert grounded.valen_decision_version == "1.5"

    explicit = ValenQwenConfig(valen_full_attention_mask="causal", valen_decision_version="1.5")
    assert explicit.valen_decision_version == "1.5"
