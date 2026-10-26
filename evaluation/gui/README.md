# GUI evaluation

Evaluate ScreenSpot, ScreenSpot-v2 and ScreenSpot-Pro separately. Use the center
of the highest-scoring predicted box and score its inclusion in the original
target box. Report overall accuracy and platform/text-icon groups; application
and domain groups are included when present. Internal bbox IoU is supplementary.
ScreenSpot-Pro uses English, positive targets and the original instruction style.

For a separate diagnostic subset, run `python -m evaluation.gui.diagnose` with
`--data-root`, `--model-path`, `--output-dir` and optional `--native` for an original
Qwen backbone. The tool compares fixed IDs at 1MP and 4MP and records predictions,
input grids and coordinate uncertainty. Native inference uses a fixed computer-use
tool prompt; spatial inference retains its trained decision prompt. These prompt
formats differ, so their gap includes prompt, readout and adaptation effects.
Diagnostic subset results never replace full benchmark scores or select a checkpoint.
Use `--model-root` instead of `--model-path` to compare all four native/spatial pairs
inside an existing eight-GPU allocation; pass its node count and GPUs per node.

```bash
python -m evaluation.gui.evaluate \
  --data-root "$VALEN_DATA_ROOT/eval/gui" \
  --model-path models/Valen-Decision-GUI-0.8B \
  --output-dir output/gui-evaluation
```

Install the project's `spatial` extra. The data root contains independent media
and `processed/{screenspot,screenspot_v2,screenspot_pro}_test.jsonl`. Use
`--benchmark` to select one file. Training checkpoints and distributed `torchrun`
launches are supported. All benchmark screenshots are excluded from training and
checkpoint selection. Invalid or missing predictions count as failures.

Sources: [ScreenSpot](https://github.com/njucckevin/SeeClick),
[ScreenSpot-v2](https://huggingface.co/datasets/OS-Copilot/ScreenSpot-v2),
[ScreenSpot-Pro](https://github.com/likaixin2000/ScreenSpot-Pro-GUI-Grounding).

Additional grounding evaluation uses the complete UI-Vision Basic/Functional/
Spatial splits (1,772/1,772/1,935), OSWorld-G refined (564, including 40 polygons
and 54 refusal targets), and MMBench-GUI L2 (3,594). Prepare them independently
with `python -m scripts.data.prepare_gui_benchmarks`; pass `--uivision-root`,
`--osworld-root`, `--mmbench-root`, `--training-root` and `--output-root`.
Source revisions and licenses are recorded in the prepared manifests. Training
overlap checks include file hashes, decoded pixels and near duplicates. Full
official splits remain intact; any overlap is disclosed alongside the score.

```bash
python -m evaluation.gui.additional \
  --data-root "$VALEN_DATA_ROOT/eval/gui/additional" \
  --model-path models/Valen-Decision-GUI-2B \
  --output-dir output/gui-additional \
  --nodes 1 --gpus-per-node 8
```

Box boundaries are inclusive; polygons use the official ray-casting rule.
All samples allow the same categorical no-object action: a no-object argmax
maps to the official negative-point refusal convention. Invalid coordinates or
nonfinite class logits remain failures, including on refusal targets. UI-Vision reports each
setting and their arithmetic mean. MMBench-GUI reports Basic/Advanced,
platform/text-icon groups, and the mean of Basic and Advanced accuracy. These
aggregates differ from pooling all records. OSWorld-G reports overall accuracy,
capability groups and refusal accuracy. Model selection uses independent
development data. Interaction trajectories and online task success are outside
this grounding evaluation.

OSWorld-G refined includes one original empty instruction; evaluation preserves
it verbatim with an explicit compiler/reader opt-in. Training validation continues
to require nonempty instructions. UI-Vision Spatial includes one original
zero-area annotation; it is retained without enlarging its scoring region.

Official sources: [UI-Vision](https://github.com/uivision/UI-Vision),
[OSWorld-G](https://github.com/xlang-ai/OSWorld-G),
[MMBench-GUI](https://github.com/open-compass/MMBench-GUI).
