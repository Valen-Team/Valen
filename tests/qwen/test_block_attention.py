"""Block-bidirectional full attention (issue #40): mask semantics and kernels."""
import pytest
import torch

from valen.modeling.qwen.block_attention import block_mask, block_attention_forward


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
    slow = block_attention._dense(Module(), q.float(), k.float(), v.float(), prefix, lengths, 0.0, dim ** -0.5)
    valid = mask.bool()
    assert (fast.float()[valid] - slow[valid]).abs().max() < 2e-2
    grad = torch.randn_like(slow)
    fast.float().mul(grad)[valid].sum().backward()
    fast_grads = [t.grad.clone() for t in (q, k, v)]
    for t in (q, k, v):
        t.grad = None
    qs, ks, vs = (t.detach().float().requires_grad_(True) for t in (q, k, v))
    block_attention._dense(Module(), qs, ks, vs, prefix, lengths, 0.0, dim ** -0.5).mul(grad)[valid].sum().backward()
    for a, b in zip(fast_grads, (qs.grad, ks.grad, vs.grad)):
        assert (a.float() - b).abs().max() < 5e-2
