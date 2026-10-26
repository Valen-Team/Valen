# Valen v1 training configs

Two-stage SFT with Qwen3.5, the Mixer decision head, `shared_state`, and FlashAttention 2. Warmup trains the Mixer only; joint SFT updates the entire backbone and Mixer (`finetuning_type: full`).

两阶段 SFT：预热阶段只训练 Mixer，联合微调阶段全参数训练。配置中的路径均相对于仓库根目录。

| Model | Warmup | Joint SFT | Joint microbatch / padded tokens per GPU |
| --- | --- | --- | --- |
| 0.8B | `0.8b_warmup.json` | `0.8b_joint.json` | 32 / 131,072 |
| 2B | `2b_warmup.json` | `2b_joint.json` | 32 / 131,072 |
| 4B | `4b_warmup.json` | `4b_joint.json` | 8 / 32,768 |

The recorded runs use **32 GPUs (4 nodes × 8 H200 GPUs)**. `tokens_per_step` is a per-GPU accumulation budget: 32,768 for warmup and 65,536 for joint SFT, giving nominal global budgets of 1,048,576 and 2,097,152 tokens. These budgets count compiled tokens; microbatch limits also account for padding. Each stage runs one epoch; `max_steps` is an upper bound. The warmup dataset is a subset reused in joint SFT.

4B joint SFT uses a smaller microbatch of 8 states and a padded token limit of 32,768 per GPU to reduce peak memory. The completed 4B warmup used 32 / 131,072. The joint accumulation budget remains 65,536 tokens per GPU. **The 4B joint run has not completed when these configs are recorded.**

## Measured time on 8 × L20Z

One epoch of the 0.8B configs, measured on a single node of 8 NVIDIA L20Z GPUs (80 GB) with only the model, data and output paths changed. The data is the image/text training set (955,000 records, 1,256,911 QA; no JevBench 30k and no video): the single-QA files regrouped with `python -m valen.data.group_questions --max-questions 4`, the multi-QA files kept as they are, 732,944 states in total. Because `tokens_per_step` is per GPU, the global batch is a quarter of the recorded 32-GPU run.

| Stage | Steps per epoch | States per step | Steady step time | Epoch, including startup |
| --- | ---: | ---: | ---: | ---: |
| Warmup (10% subset) | 124 | ≈590 | ≈1.2 s | ≈5 min |
| Joint SFT | 616 | ≈1,190 | ≈4.4 s | ≈1.2 h |

- Environment: PyTorch 2.10.0+cu128 (cuDNN 9.10), Transformers 5.14.1, flash-attn 2.8.3, flash-linear-attention 0.4.2. The times are extrapolated from 150 joint and 40 warmup steps. Before the kernels warm up, some steps take ≈28 s instead of ≈4–5 s: 15 of the first 150 joint steps, none in steps 126–150. The estimate includes them. A joint checkpoint is 9.6 GB and takes ≈10 s to save; peak memory is 37.8 GB per GPU.
- With PyTorch 2.9.x and cuDNN 9.8–9.14, the BF16 Conv3d in the vision patch embedding falls back to `aten::slow_conv_dilated3d` (upstream [pytorch#166122](https://github.com/pytorch/pytorch/issues/166122), [pytorch#174051](https://github.com/pytorch/pytorch/issues/174051)). On the same node this gives about 4.3 states/s, roughly 47 h for the joint epoch, estimated from a 20,000-record run. PyTorch 2.6 (the version pinned in `pyproject.toml`), PyTorch 2.10 or cuDNN 9.15+ avoids the fallback. With 2.9.x, converting that weight to `channels_last_3d` restores the cuDNN path and the speed in the table.

8 卡 L20Z 实测：图文数据（73 万个 state）上，0.8B 的预热约 5 分钟，联合全参数训练约 1.2 小时。PyTorch 2.9.x 搭配 cuDNN 9.8–9.14 时，视觉 patch embedding 的 Conv3d 会退到慢路径，耗时约为 40 倍。

## Run

From the repository root, prepare the matching base model at `models/Qwen3.5-{0.8B,2B,4B}` and your datasets at `data/valen1/{warmup,train}.jsonl`, or edit those paths in the configs. Relative media paths resolve against the JSONL's parent directory. Dataset contents and machine-specific launch settings are supplied separately.

Set `NODE_RANK` (0–3), `MASTER_ADDR` (node 0's reachable address), and `MASTER_PORT` consistently for all four nodes. Run each stage on all nodes, waiting for warmup to finish before starting joint SFT:

```bash
VALEN_SIZE=2b  # 0.8b / 2b / 4b

torchrun --nnodes=4 --nproc_per_node=8 --node_rank="$NODE_RANK" \
  --rdzv_backend=c10d --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
  --rdzv_id="valen1-${VALEN_SIZE}-warmup" \
  -m valen.train --config "configs/train/qwen/valen1/${VALEN_SIZE}_warmup.json"

torchrun --nnodes=4 --nproc_per_node=8 --node_rank="$NODE_RANK" \
  --rdzv_backend=c10d --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
  --rdzv_id="valen1-${VALEN_SIZE}-joint" \
  -m valen.train --config "configs/train/qwen/valen1/${VALEN_SIZE}_joint.json" \
  --initialize "output/valen1/${VALEN_SIZE}/warmup/latest"
```

To resume an interrupted joint run, replace `--initialize ...` with `--resume "output/valen1/${VALEN_SIZE}/joint/latest"`. Keep the same world size and data/config paths as the checkpoint. For an existing checkpoint created with different absolute paths, use its original runtime config with the current microbatch limits, or use `--initialize` for a new stage with reset optimizer and progress.

断点续训使用 `--resume`；切换阶段使用 `--initialize`。公开配置已替换本地路径，不能直接作为旧训练任务的原路径断点配置。
