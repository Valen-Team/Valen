# Decision Index public evaluation

**English** · [简体中文](README_zh.md)

Run published Valen models with the official
[Decision Index reproduction kit](https://github.com/apolinario/decision-index).
The adapter uses the native decision head and shared-state inference. The upstream
runner, response validation, frozen suites and scorers remain unchanged.

## Install

From the Valen repository root:

```bash
python -m pip install -e ".[decision-index]"
python -m evaluation.decision_index --help
```

The optional dependency pins the upstream kit to
[`f5a6e4f`](https://github.com/apolinario/decision-index/tree/f5a6e4fd85fed05e77878dfc4966d0568bfb45ec),
the revision used for our public evaluations. The module command forwards to that
kit and supplies its missing `hashlib.file_digest` API on Python 3.10.

## Prepare the frozen suite

Prepare the official files using the upstream
[rebuild/import instructions](https://github.com/apolinario/decision-index/tree/f5a6e4fd85fed05e77878dfc4966d0568bfb45ec#quickstart-local).
Rebuilding also requires the kit's `rebuild` extra. Keep the prepared suite in
`data/decision-index/suite-0.3`, or select another directory below. Its files are
`selected-rows.jsonl.gz`, `added-rows.jsonl.gz`, `gsm8k-rows.jsonl.gz`,
`excluded-questions.json` and `manifest.json`.

```bash
python -m evaluation.decision_index suite verify \
  --edition 0.3 --dir data/decision-index/suite-0.3
```

Check the content, exclusion and subset matches. The evaluation pipeline verifies
the frozen data before loading the model. Data and results stay outside Git.

## Evaluate

```bash
python -m evaluation.decision_index pipeline \
  --edition 0.3 \
  --suite-dir data/decision-index/suite-0.3 \
  --engine evaluation.decision_index.engine:ValenEngine \
  --model Valen-Team/Valen-2B \
  --option device=cuda \
  --option attention=sdpa \
  --out output/decision-index/0.3/Valen-2B \
  --compact
```

- `--model`: a complete published model directory or Hugging Face repository,
  including `Valen-Team/Valen-0.8B`, `Valen-Team/Valen-2B` and
  `Valen-Team/Valen-4B`. Hub models download to the standard Hugging Face cache.
- `--option revision=<commit>`: pin the Hub model revision; default `main`.
- `--option device=cpu` or `device=cuda:1`: choose a different device.
- `--option attention=eager|sdpa|flash_attention_2`: choose one backend.
  Flash Attention requires a compatible installation and CUDA GPU; see
  [attention settings](../../docs/configuration.md).

For the CLEF-compatible **0.2.1 public suite**, change `--edition` to `0.2.1`,
point `--suite-dir` at its separately prepared suite, and use a separate output
directory. It needs the base and added rows, exclusions and manifest, without
`gsm8k-rows.jsonl.gz`. Both editions retain their own scoring rules.

Re-running the same command resumes the upstream results file and retries runtime
errors. A quick pipeline check can add `--limit 10`; its partial score is not a
complete benchmark result.

## ONNX CPU evaluation

Published ONNX models use `ValenONNXEngine` with the same upstream runner and scorers. Use Python 3.11+ and run from the Valen repository root. Inference runs through the standalone SDK without loading a PyTorch model. Use the frozen suite prepared and verified above.

```bash
python -m pip install \
  "valen-sdk[onnx]==0.2.0" \
  "decision-index @ git+https://github.com/apolinario/decision-index.git@f5a6e4fd85fed05e77878dfc4966d0568bfb45ec"
python -m evaluation.decision_index pipeline \
  --edition 0.3 \
  --suite-dir data/decision-index/suite-0.3 \
  --engine evaluation.decision_index.onnx_engine:ValenONNXEngine \
  --model Valen-Team/Valen-0.8B-ONNX \
  --option variant=fp32 \
  --option threads=4 \
  --out output/decision-index/0.3/Valen-0.8B-ONNX-fp32 \
  --compact
```

`--model` also accepts the 2B or 4B ONNX repository or a local model directory. Use `--option variant=q8-mlp` for CPU MLP quantization and `--option threads=4` to set CPU threads. Once fully cached, add `--option offline=true` to prevent downloads; `--option revision=<commit>` selects a model revision. Keep separate output directories for each model and variant.

This adapter uses the native decision head and `shared_state`, retaining the upstream questions and candidate order without truncation. It is a CPU evaluation adapter; see [ONNX & SDKs](../../docs/onnx.md) for in-app CUDA, images/video and browser inference.


## Inputs, precision and outputs

The adapter keeps every question ID, option and insertion order. Strings remain
unchanged; structured state, instructions and option descriptions use the
upstream compact UTF-8 JSON renderer. A null option description uses its option
key. Only state and questions reach Valen; expected labels stay with the scorer.

The model inherits its saved head and maximum length. Shared-state inference and
temperature 1.0 are fixed. Published parameter precision is preserved; CUDA uses
BF16 backbone autocast and CPU disables autocast. Actual parameter
and autocast dtypes are recorded in `environment.json`. Overlong inputs and
reserved tokenizer control tokens are reported as `Unsupported`, with no
truncation or option filtering. Other exceptions remain runtime errors.

The upstream pipeline writes `results.jsonl`, `status.json`, `environment.json`,
`benchmark-summary.json`, `index.json` and `scores.json`. `--compact` omits source
payloads from the prediction log. Loading and warmup are outside request timing;
compilation, forward and native answer conversion are included.

These commands calculate the **public index**. HF's Full score also requires
private evaluations by the maintainers; the AA Decision Index and the HF Vision
board are separate evaluations. Official latency is measured by the maintainers
on their reference hardware. See the upstream
[submission instructions](https://github.com/apolinario/decision-index#submitting-a-model-to-the-leaderboard).
