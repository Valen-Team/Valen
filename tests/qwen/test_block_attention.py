"""Block-bidirectional full attention (issue #40): mask semantics and kernels."""
import pytest
import torch

from valen.modeling.qwen.block_attention import block_mask, block_attention_forward, segment_mask


class Module(torch.nn.Module):
    is_causal = True


def reference(prefix, lengths, length):
    """Spell out the rule token by token. / 逐 token 写出规则。"""
    batch = len(prefix)
    mask = torch.zeros(batch, 1, length, length, dtype=torch.bool)
    for b in range(batch):
        p, n = int(prefix[b]), int(lengths[b])
        for q in range(length):
            for k in range(n):
                mask[b, 0, q, k] = (k < p) if q < p else True
    return mask


def test_block_mask_rule():
    prefix, lengths = torch.tensor([3, 1, 5]), torch.tensor([6, 4, 5])
    assert torch.equal(block_mask(prefix, lengths, 7), reference(prefix, lengths, 7))


def test_prefix_never_sees_suffix_and_suffix_is_bidirectional():
    mask = block_mask(torch.tensor([4]), torch.tensor([7]), 7)[0, 0]
    assert not mask[:4, 4:].any()                 # prefix cannot read questions
    assert mask[4:, :7].all()                     # questions read everything
    assert mask[:4, :4].all()                     # state block is bidirectional too


def test_matches_transformers_block_sequence_ids():
    masking = pytest.importorskip("transformers.masking_utils")
    if not hasattr(masking, "blockwise_overlay"):
        pytest.skip("Transformers without block_sequence_ids")
    prefix, length = 3, 6
    ids = torch.tensor([[0] * prefix + [1] * (length - prefix)])
    rule = masking.or_masks(masking.causal_mask_function, masking.blockwise_overlay(ids))
    t = torch.tensor
    native = torch.tensor([[[[bool(rule(t(0), t(0), t(q), t(k))) for k in range(length)] for q in range(length)]]])
    ours = block_mask(torch.tensor([prefix]), torch.tensor([length]), length)
    assert torch.equal(native, ours)


def test_suffix_change_does_not_change_prefix_outputs():
    torch.manual_seed(0)
    q, k, v = (torch.randn(1, 2, 8, 4) for _ in range(3))
    prefix = torch.tensor([5])
    base, _ = block_attention_forward(Module(), q, k, v, None, valen_prefix_lengths=prefix)
    k2, v2 = k.clone(), v.clone()
    k2[:, :, 5:] += 1.0
    v2[:, :, 5:] += 1.0
    changed, _ = block_attention_forward(Module(), q, k2, v2, None, valen_prefix_lengths=prefix)
    assert torch.allclose(base[:, :5], changed[:, :5])
    assert not torch.allclose(base[:, 5:], changed[:, 5:])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FlashAttention path needs CUDA")
def test_flash_matches_dense_with_padding_and_gqa():
    pytest.importorskip("flash_attn")
    from valen.modeling.qwen import block_attention
    torch.manual_seed(0)
    batch, heads, kv_heads, length, dim = 3, 8, 2, 37, 64
    q = torch.randn(batch, heads, length, dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(batch, kv_heads, length, dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(batch, kv_heads, length, dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    lengths = torch.tensor([37, 20, 30], device="cuda")
    prefix = torch.tensor([10, 20, 4], device="cuda")
    mask = (torch.arange(length, device="cuda")[None] < lengths[:, None]).long()
    fast, _ = block_attention_forward(Module(), q, k, v, mask, valen_prefix_lengths=prefix, scaling=dim ** -0.5)
    slow = block_attention._dense(q.float(), k.float(), v.float(), block_mask(prefix, lengths, length), 0.0, dim ** -0.5)
    valid = mask.bool()
    assert (fast.float()[valid] - slow[valid]).abs().max() < 2e-2
    grad = torch.randn_like(slow)
    fast.float().mul(grad)[valid].sum().backward()
    fast_grads = [t.grad.clone() for t in (q, k, v)]
    for t in (q, k, v):
        t.grad = None
    qs, ks, vs = (t.detach().float().requires_grad_(True) for t in (q, k, v))
    block_attention._dense(qs, ks, vs, block_mask(prefix, lengths, length), 0.0, dim ** -0.5).mul(grad)[valid].sum().backward()
    for a, b in zip(fast_grads, (qs.grad, ks.grad, vs.grad)):
        assert (a.float() - b).abs().max() < 5e-2


def test_prefix_lm_suffix_is_causal():
    mask = block_mask(torch.tensor([3]), torch.tensor([6]), 6, causal_suffix=True)[0, 0]
    assert mask[:3, :3].all() and not mask[:3, 3:].any()
    for q in range(3, 6):
        assert mask[q, :q + 1].all() and not mask[q, q + 1:].any()


def test_question_blocks_isolate_questions():
    # prefix(3) | glue(1) | q1(2) | q2(2) | glue(1) | q1 readout(1) | q2 readout(1) | pad(1)
    segments = torch.tensor([[-1, -1, -1, 0, 1, 1, 2, 2, 0, 1, 2, -2]])
    mask = segment_mask(segments, torch.tensor([3]), torch.tensor([11]))[0, 0]
    assert mask[:3, :3].all() and not mask[:3, 3:].any()
    q1, q2 = [4, 5, 9], [6, 7, 10]
    for a in q1:
        assert mask[a, :4].all() and mask[a, q1].all() and not mask[a, q2].any() and mask[a, 8]
    for a in q2:
        assert not mask[a, q1].any() and mask[a, q2].all()
    assert not mask[:, 11].any()


def test_prefix_lm_matches_dense_on_cpu():
    torch.manual_seed(0)
    q, k, v = (torch.randn(2, 2, 7, 4) for _ in range(3))
    prefix, mask2d = torch.tensor([3, 2]), torch.tensor([[1] * 7, [1] * 5 + [0] * 2])
    out, _ = block_attention_forward(Module(), q, k, v, mask2d, valen_prefix_lengths=prefix, valen_mask_mode="prefix_lm")
    k2 = k.clone(); k2[:, :, 6] += 3  # last token changes nothing before it
    out2, _ = block_attention_forward(Module(), q, k2, v, mask2d, valen_prefix_lengths=prefix, valen_mask_mode="prefix_lm")
    assert torch.allclose(out[0, :6], out2[0, :6])


def test_bbox_blocks_isolate_only_localization_questions():
    """bbox questions keep their own block; decision questions share one."""
    from types import SimpleNamespace
    from valen.modeling.qwen.block_attention import forward_kwargs
    # prefix(2) | glue(1) | q1 choice(2) | q2 bbox(2) | q3 choice(1) | readouts(3)
    segments = torch.tensor([-1, -1, 0, 1, 1, 2, 2, 3, 1, 2, 3])
    branch = SimpleNamespace(context_span=(0, 2), segments=segments, bbox_segments=(2,))
    kwargs = forward_kwargs("bbox_blocks", [branch], "cpu")
    merged = kwargs["valen_segments"][0]
    # Decision questions 1 and 3 are merged into block 1; the bbox question keeps block 2.
    assert merged.tolist() == [-1, -1, 0, 1, 1, 2, 2, 1, 1, 2, 1]
    mask = segment_mask(kwargs["valen_segments"], torch.tensor([2]), torch.tensor([11]))[0, 0]
    bbox_rows = [5, 6, 9]
    decision_rows = [3, 4, 7, 8, 10]
    for a in bbox_rows:
        assert not mask[a, decision_rows].any() and mask[a, bbox_rows].all()
    for a in decision_rows:
        assert mask[a, decision_rows].all() and not mask[a, bbox_rows].any()
        assert mask[a, :2].all() and mask[a, 2]
