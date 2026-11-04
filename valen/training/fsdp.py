"""FSDP2 sharding for large Qwen backbones (Valen-next 27B).

只对语言塔做 FSDP2 分片（每个 decoder 层一个单元），视觉塔与决策头保持复制，
梯度仍由 ``synchronize_gradients`` 求和。所有 rank 每步必须执行相同次数的前向/反向，
否则分片的 all-gather / reduce-scatter 会失配；runner 用零损失的空单元补齐。

Only the language tower is sharded (one FSDP unit per decoder layer); the vision
tower and decision heads stay replicated and are summed by ``synchronize_gradients``.
Every rank must run the same number of forward/backward passes per step, so the
runner pads with zero-loss dummy units.
"""
import json
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor


def is_sharded(parameter):
    return isinstance(parameter, DTensor)


def shard_language_model(model, context):
    """Shard ``model.backbone.language_model`` in place and move the model to GPU."""
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

    policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    language = model.backbone.language_model
    for layer in language.layers:
        fully_shard(layer, mp_policy=policy, reshard_after_forward=True)
    fully_shard(language, mp_policy=policy, reshard_after_forward=True)
    # Losses already divide by the GLOBAL question/state count: sum, never average.
    # 损失已按全局数量归一化：梯度求和而非平均。
    for module in [*language.layers, language]:
        module.set_gradient_divide_factor(1.0)
    model.to(context.device)
    # FP32 master weights for every trainable tensor; compute stays BF16.
    # 可训练参数保存 FP32 主权重，计算仍为 BF16。
    for parameter in model.parameters():
        if parameter.requires_grad and parameter.dtype != torch.float32:
            parameter.data = parameter.data.float()
    return model


@torch.no_grad()
def broadcast_replicated(model, context):
    if context.world_size > 1:
        for parameter in model.parameters():
            if parameter.requires_grad and not is_sharded(parameter):
                dist.broadcast(parameter.data, src=0)


@torch.no_grad()
def clip_gradients(model, context, max_norm):
    """Global L2 norm over sharded and (already summed) replicated gradients."""
    sharded, replicated = [], []
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        (sharded if is_sharded(parameter.grad) else replicated).append(parameter.grad)
    device = torch.device(context.device)
    local = torch.zeros((), device=device, dtype=torch.float64)
    for grad in sharded:
        local += grad.to_local().double().square().sum()
    if context.world_size > 1:
        dist.all_reduce(local)
    for grad in replicated:
        local += grad.double().square().sum()
    norm = local.sqrt()
    if not torch.isfinite(norm):
        raise FloatingPointError("Non-finite gradient norm")
    coefficient = (max_norm / (norm + 1e-6)).clamp(max=1.0)
    for grad in sharded:
        grad.to_local().mul_(coefficient.to(grad.dtype))
    for grad in replicated:
        grad.mul_(coefficient.to(grad.dtype))
    return float(norm)


def group_norms(groups, context):
    result = {}
    for group in groups:
        local = torch.zeros((), device=torch.device(context.device), dtype=torch.float64)
        replicated = torch.zeros_like(local)
        found = False
        for parameter in group["params"]:
            if parameter.grad is None:
                continue
            found = True
            if is_sharded(parameter.grad):
                local += parameter.grad.to_local().double().square().sum()
            else:
                replicated += parameter.grad.double().square().sum()
        flag = torch.tensor([float(found)], device=local.device)
        if context.world_size > 1:
            dist.all_reduce(local)
            dist.all_reduce(flag)
        if flag.item():
            result[group["name"]] = float((local + replicated).sqrt())
    return result


def save_sharded(path, model, optimizer, extra, context):
    """Distributed checkpoint of model + optimizer plus rank-0 JSON metadata."""
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import get_state_dict

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    model_state, optimizer_state = get_state_dict(model, optimizer)
    dcp.save({"model": model_state, "optimizer": optimizer_state}, checkpoint_id=str(path / "dcp"))
    if context.primary:
        (path / "trainer_state.json").write_text(json.dumps(extra, indent=2, default=str) + "\n", encoding="utf-8")
    context.barrier()


def load_sharded(path, model, optimizer=None):
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict, StateDictOptions

    path = Path(path)
    options = StateDictOptions(strict=optimizer is not None)
    if optimizer is None:
        from torch.distributed.checkpoint.state_dict import get_model_state_dict, set_model_state_dict
        state = {"model": get_model_state_dict(model)}
        dcp.load(state, checkpoint_id=str(path / "dcp"))
        set_model_state_dict(model, state["model"], options=options)
    else:
        model_state, optimizer_state = get_state_dict(model, optimizer)
        state = {"model": model_state, "optimizer": optimizer_state}
        dcp.load(state, checkpoint_id=str(path / "dcp"))
        set_state_dict(model, optimizer, model_state_dict=state["model"], optim_state_dict=state["optimizer"],
                       options=options)
    return json.loads((path / "trainer_state.json").read_text(encoding="utf-8"))


def full_state_dict(model):
    """Gather full CPU tensors on every rank (rank 0 writes). / 收集完整权重到 CPU。"""
    from torch.distributed.checkpoint.state_dict import get_model_state_dict, StateDictOptions
    return get_model_state_dict(model, options=StateDictOptions(full_state_dict=True, cpu_offload=True))
