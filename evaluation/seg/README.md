# Seg evaluation

RefCOCO and RefCOCO+ use UNC val/testA/testB; RefCOCOg uses UMD val/test. PhraseCut
uses val_miniv for development and its complete test split for final evaluation.
Report mean per-expression IoU and cumulative intersection/union. Precision at
IoU > 0.5 is also recorded. All masks are evaluated at original image resolution.

```bash
python -m evaluation.seg.evaluate \
  --data-root "$VALEN_DATA_ROOT/eval/seg" \
  --model-path models/Valen-Decision-Seg-0.8B \
  --output-dir output/seg-evaluation
```

Install the project's `spatial` extra. The data root contains
`processed/{benchmark}_{split}.jsonl` and independent image/mask assets. Genuine
RLE, polygon or binary PNG targets are supported. Use `--benchmark` to select one
file; training checkpoints and distributed `torchrun` launches are supported.
Masks are scored in memory without accumulating full-resolution PNG predictions.

Sources: [REFER](https://github.com/lichengunc/refer),
[PhraseCut](https://github.com/ChenyunWu/PhraseCutDataset).

ReasonSeg preparation: `python -m pip install -e '.[spatial,reasonseg]'`, then
`python -m scripts.data.prepare_spatial_reasonseg --source-root "$REASONSEG_SOURCE" --data-root "$VALEN_DATA_ROOT"`.
Use the same evaluator with `--data-root "$VALEN_DATA_ROOT/eval/seg/reasonseg"`.
The [pinned LISA protocol](https://github.com/JIA-Lab-research/LISA/tree/3cb2d4301f1af4691bd4f3938335ef06e76f155a)
sorts polygons by rasterized area, draws large regions first, ignores `flag` annotations
and excludes label-255 regions from both losses and metrics. Original masks remain at
source resolution. Report foreground mean IoU (`gIoU`, also exposed as `mIoU`) and
cumulative `cIoU`; validation/test use the first original instruction per image.
Training retains all unique original paraphrases in `seg/processed/reasonseg_train.jsonl`.
Original text is retained in metadata; fixed question/output formatting does not
create extra paraphrases or samples.

`configs/train/qwen/{det,seg}/reasonseg_sft.json` supports each task through
`VALEN_MODEL_PATH`, `VALEN_DATA_ROOT`, `VALEN_RUN_DIR` and `VALEN_STAGE=warmup` or `joint`.
Launch Det stages with sixteen ranks and Seg stages with eight; initialize the
joint stage from its warmup best checkpoint. Official validation selects checkpoints;
test remains held out. Keep these runs separate from already sealed LoRA/Full comparisons.
Empty original masks remain in the dataset and follow the official empty-union rule;
fully ignored pixels contribute no mask or presence loss.
