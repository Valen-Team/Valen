# Task evaluation

This directory contains environments for evaluating complete decision sequences. For labeled JSONL questions and probability metrics, use `python -m valen.evaluate`; see the [script guide](../scripts/README.md).

The [JevBench evaluator](jevbench/README.md) runs local Valen checkpoints on a
frozen 231-task public snapshot, with upstream scoring, calibration and latency
reports. Its local experiment files live in the Git-ignored `JevBench-exp/`.

The [Decision Index adapter](decision_index/README.md) connects published Valen
models to the upstream frozen 0.3 and 0.2.1 public suites, runner and scorers.
It accepts Hugging Face model IDs or local complete-model directories.

The [Sokoban benchmark](sokoban/README.md) includes a deterministic environment, an exact solver, synthetic data generation and validation, full-game evaluation, result comparison, and trajectory replay. It supports Valen, the original Qwen generation model, a random policy, and an oracle sanity check.

[game-v4 and arcade-v2](game/README.md) add labeled action scoring and complete-game evaluation for Maze, FrozenLake, Sokoban, LaneRacer and ParkourRunner. The new `game-v4/sokoban` and legacy `sokoban-v1/sokoban` results are reported separately.

Run commands from the repository root after installing the project. Generated data and results are excluded from Git. The evaluators are included in the Python package and also run outside a checkout with explicit data/model/output paths; default paths are relative to the working directory. Preparation and training scripts run from the checkout.

Spatial decision benchmarks are provided by [Det](det/README.md),
[Seg](seg/README.md) and [GUI](gui/README.md). Their shared runner accepts external
data/model/output paths and supports distributed evaluation without bundling
datasets, local machine settings or historical prediction files.
Spatial latency values span the per-record evaluation loop: image processing,
model forward, decoding and metric updates. Reports also record hardware, pixel
budget and batch size per rank.

Save a few fixed, label-independent qualitative examples with
`python -m evaluation.examples --task gui --data-root "$VALEN_DATA_ROOT/eval/gui" --model-path "$MODEL_PATH" --output-dir output/gui-examples`.
Det defaults to one example from each RefCOCO-family source and ReasonSeg; Seg
and GUI use three benchmarks each. Samples are selected by hashed ID. The JSON and compact image sheet supplement full scores.

## Scope

Reusable evaluators and bounded spatial diagnostics are included. Diagnostic tools
use fixed sample IDs and keep their results separate from complete benchmark scores.
Machine-specific job settings, alternate test-set variants, historical reports and
model checkpoints are not bundled.
