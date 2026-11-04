"""Synchronous data parallelism for unequal numbers of question backwards.

Gradients are summed only after local accumulation, so an idle rank and a
text-only rank participate in the same collectives as a visual/Score rank.
"""
from dataclasses import dataclass
from datetime import timedelta
import os
import random
import torch
import torch.distributed as dist


@dataclass
class Distributed:
    rank: int = 0
    world_size: int = 1
    device: str = "cpu"

    @property
    def primary(self):
        return self.rank == 0

    def sum(self, values):
        tensor = torch.tensor(values, dtype=torch.float64, device=self.device)
        if self.world_size > 1:
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return tensor.tolist()

    def gather(self, value):
        if self.world_size == 1:
            return [value]
        values = [None] * self.world_size
        dist.all_gather_object(values, value)
        return values

    def barrier(self):
        if self.world_size > 1:
            dist.barrier()


def initialize(device="cuda"):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if str(device).startswith("cuda"):
        index = int(os.environ["LOCAL_RANK"]) if world_size > 1 else torch.device(device).index
        if index is None:
            index = torch.cuda.current_device()
        torch.cuda.set_device(index)
        device = f"cuda:{index}"
    if world_size > 1:
        if not dist.is_initialized():
            dist.init_process_group("nccl" if str(device).startswith("cuda") else "gloo",
                                    timeout=timedelta(minutes=10))
        rank, world_size = dist.get_rank(), dist.get_world_size()
    return Distributed(rank, world_size, str(device))


def close():
    if dist.is_initialized():
        dist.destroy_process_group()


def epoch_shard(size, seed, epoch, rank, world_size):
    order = list(range(size))
    random.Random(seed + epoch).shuffle(order)
    # No padding or dropped tail: each state belongs to exactly one rank.
    return order[rank::world_size]


@torch.no_grad()
def broadcast_trainable(model, context):
    if context.world_size > 1:
        for parameter in model.parameters():
            if parameter.requires_grad:
                dist.broadcast(parameter, src=0)


@torch.no_grad()
def synchronize_gradients(model, context, bucket_bytes=16 * 1024 * 1024):
    if context.world_size == 1:
        return
    from torch.distributed.tensor import DTensor
    # FSDP-sharded tensors are reduce-scattered by FSDP itself. / 分片参数由 FSDP 归约。
    parameters = [p for p in model.parameters() if p.requires_grad and not isinstance(p, DTensor)]
    used = torch.tensor([p.grad is not None for p in parameters], dtype=torch.int32, device=context.device)
    dist.all_reduce(used, op=dist.ReduceOp.MAX)
    # Globally unused parameters retain grad=None, avoiding AdamW weight decay.
    active = [p for p, flag in zip(parameters, used.tolist()) if flag]
    buckets, bucket, size = [], [], 0
    for p in active:
        nbytes = p.numel() * p.element_size()
        if bucket and (size + nbytes > bucket_bytes or p.dtype != bucket[0].dtype):
            buckets.append(bucket)
            bucket, size = [], 0
        bucket.append(p)
        size += nbytes
    if bucket:
        buckets.append(bucket)
    for bucket in buckets:
        flat = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1) for p in bucket])
        # Losses already divide by the GLOBAL state count. Do not divide again.
        dist.all_reduce(flat, op=dist.ReduceOp.SUM)
        offset = 0
        for p in bucket:
            p.grad = flat[offset:offset + p.numel()].view_as(p)
            offset += p.numel()
