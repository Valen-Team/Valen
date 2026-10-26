# Det evaluation

For multiple Jev credentials, launch `python -m evaluation.det.jev_keys` with
`--credential-env-file`, `--key-envs` (environment variable names), and
`--credential-status`, followed by the usual Jev evaluation arguments. Requests
use sequential round robin; rejected or limited credentials fail over to another
key. HTTP 5xx responses also try the remaining keys, without disabling them;
each request makes at most one attempt per configured key. Credentials never
enter results. The sealed evaluator, request format and
existing prediction cache remain unchanged, allowing evaluation to resume.

Generate verified full-split reports with `python -m evaluation.det.jev_report`:
pass `--run-dir`, `--data-root`, `--output-dir`, optional `--workers-root`, and
`--watch` for background updates. Add `--registry` and `--model-root` to compare
accepted 0.8B/2B/4B Det LoRA models. Reports verify exact prediction-ID coverage,
model version, source fingerprints and recomputed scores. Incomplete splits
remain empty; a full completion marker requires all 59,441 queries across eleven
splits. API latency includes networking and retries.

The current Det series trains on 100k RefCOCO, RefCOCO+, RefCOCOg and LISA/ReasonSeg
examples. COCO category-detection samples and benchmarks are retired.
RefCOCO/RefCOCO+ use UNC val/testA/testB; RefCOCOg uses UMD val/test.
Report top-1 localization accuracy at IoU >= 0.5 for each complete split.

```bash
python -m pip install -e '.[spatial]'
python -m scripts.data.prepare_det_referring \
  --data-root "$VALEN_DATA_ROOT" --source-root "$REFCOCO_SOURCE"
# Stop obsolete Det readers before publishing the staged data.
python -m scripts.data.prepare_det_referring \
  --data-root "$VALEN_DATA_ROOT" --source-root "$REFCOCO_SOURCE" --activate
python -m evaluation.det.evaluate \
  --data-root "$VALEN_DATA_ROOT/eval/det" \
  --model-path models/Valen-Decision-Det-0.8B \
  --output-dir output/det-evaluation
```

Preparation uses pinned original instructions, content and target deduplication,
image-balanced sampling, and at most 12 referring records per image. All 1,324
ReasonSeg train instructions are retained; RefCOCO-family supplies the remaining
98,676 records. Development contains 512 RefCOCO-family val instructions and all
200 ReasonSeg val instructions. Training and evaluation images remain disjoint.
RefCOCO retains its underlying COCO images as independent task-local files.

The evaluator defaults to the eight RefCOCO-family splits and never implicitly
adds COCO. Use `--benchmark` to select one split. Data/model/output locations are
external CLI arguments; media references are relative to the data root.
Training checkpoints are accepted with optional `--base-model-path`.
Launch with `torchrun` for distributed scoring. Det training uses 16 GPUs and
gradient accumulation 2 (global batch 32); LoRA/Full comparisons share this protocol.
Results are compact JSON.
For inference, put the object description in `instructions`, keep
`categories=["target"]`, and set `meta.referring=true` to return the top box.
ReasonSeg also sets `meta.allow_empty_target=true` for every record, including
positive records, so its presence rule does not disclose the ground truth.

Prepare ReasonSeg with
`python -m scripts.data.prepare_spatial_reasonseg --source-root "$REASONSEG_SOURCE" --data-root "$VALEN_DATA_ROOT"`.
Run its complete val/test supplement on a single device with
`python -m evaluation.det.evaluate --data-root "$VALEN_DATA_ROOT/eval/det/reasonseg" --model-path "$MODEL_PATH" --output-dir output/det-reasonseg`.
The pipeline runs the same supplement across its existing sixteen-GPU group.
For that coordinated entrypoint, each of the four logical nodes runs
`python -m evaluation.reasonseg --task det --data-root "$VALEN_DATA_ROOT/eval/det/reasonseg" --model-path "$MODEL_PATH" --output-dir output/det-reasonseg --nodes 4 --gpus-per-node 4 --launch-id "$RUN_ID-reasonseg"`.
The worker supplies `KUBEBRAIN_REPLICA` (0–3) and `KUBEBRAIN_REPLICA_TOTAL=4`;
use a fresh launch ID for each invocation.
Targets are tight boxes of the valid target-mask union; ignored pixels are excluded.
Original ReasonSeg text is preserved in metadata. Short noun phrases use a fixed
question wrapper, and every instruction receives the same bbox-output request;
this formatting does not add training samples or paraphrases.
This is a **mask-derived ReasonSeg bbox adaptation**, scored by top-1 Acc@IoU>=0.5,
not the official segmentation metric. Val/test use each image's first original
instruction. Empty targets are retained; background winners produce no box under
the same presence rule for every sample. Sources: [REFER](https://github.com/lichengunc/refer)
and [LISA's fixed source protocol](https://github.com/JIA-Lab-research/LISA/tree/3cb2d4301f1af4691bd4f3938335ef06e76f155a).

Legacy COCO utilities remain available only for archived protocols, selected
explicitly with `--coco-protocol conditional` or `standard`. Their scores are not
part of the current referring-only model reports.

## Jev API comparison

Use the official [Box](https://docs.typesafe.ai/primitives/box),
[Noul](https://docs.typesafe.ai/primitives/noul) and
[image](https://docs.typesafe.ai/images) interfaces with the pinned `jev-1.14.0`
model. Read `JEV_API_KEY` from the environment, or a local ignored env file:

```bash
python -m evaluation.det.jev --data-root "$VALEN_DATA_ROOT/eval/det" \
  --env-file .env --output-dir output/jev-refcoco-pilot --limit-per-benchmark 4
# Separate output for all eleven splits (59,441 queries):
python -m evaluation.det.jev --data-root "$VALEN_DATA_ROOT/eval/det" \
  --env-file .env --output-dir output/jev-refcoco-full --limit-per-benchmark 0
```

The pilot is explicitly labeled as sampled. Full scoring covers the eight RefCOCO
splits, ReasonSeg val/test and GroundingME test. It uses every original instruction
and one native Box per image/instruction, with no Box-confidence filter.
ReasonSeg/GroundingME additionally ask a native Noul presence question for every
record, including positives; `noul >= 0.5` keeps the box. This fixed threshold is
never fitted on a test split. GroundingME retains all four dimensions/subgroups.
Ground-truth boxes and annotation metadata never enter API requests. Client-side
image resizing starts from the Det Qwen grid and 262,144-pixel budget, then respects
the service's 280–1568 side limits with centered white padding as needed. Predictions
are mapped back through that recorded transform before scoring; Jev's internal
preprocessing is unspecified. Record normalized xyxy IoU, model version, latency
and token usage. Successful predictions are reused on restart. Transport and
authentication failures stop the run and never become zero benchmark scores.
Transient transport errors, rate limits and server errors receive bounded retries.
Each output directory is locked against concurrent writers and reused only with
the same model, data, request protocol and evaluation code.

## GroundingME

All four Det LoRA sizes and the three Full controls also run the complete
[GroundingME test split](https://huggingface.co/datasets/lirang04/GroundingME)
(1,005 queries). Keep this supplement separate from RefCOCO and ReasonSeg.

```bash
python -m scripts.data.prepare_groundingme \
  --source-root "$GROUNDINGME_PARQUET_ROOT" --data-root "$VALEN_DATA_ROOT"
python -m evaluation.det.groundingme \
  --data-root "$VALEN_DATA_ROOT/eval/det/groundingme" \
  --model-path "$MODEL_PATH" --output-dir output/det-groundingme
```

Preparation pins dataset revision `78e3c7974b2b1db0ea52266969e40f664a38e330`
and the [official scorer/prompt](https://github.com/lirang04/GroundingME/blob/6867f7a045596d29cd77a14a01d22d4a68d64fb6/evaluate.py).
Preserve the original descriptions, viewer-relative spatial definition, native
image bytes and empty targets. Report sample-weighted Acc@IoU>=0.5 overall,
each of Discriminative/Spatial/Limited/Rejection, and every available L2 group;
also retain Acc@0.75/0.9 and mean IoU for auditing. Explicit empty predictions
match rejection targets; they fail positive targets. Every query uses the same
existing background-winner presence rule, without threshold fitting.

Valen uses its structured head rather than generating JSON. Its normalized xyxy
coordinates have a fixed interpretation; the adapter does not use ground truth
to choose a coordinate format. Labels and task dimensions are excluded from
inference metadata. Record the model's pixel budget, exact data/code/export
hashes and complete counts. No test examples are used for training or selection.
`torchrun` can distribute inference; a reserved group can instead supply
`--nodes`, `--gpus-per-node` and a fresh `--launch-id` on every node.
Keep only compact metrics and the predictions needed to audit the score.
Dataset use follows the original SA-1B/HR-Bench terms (research use only);
the repository license does not grant dataset redistribution rights.
