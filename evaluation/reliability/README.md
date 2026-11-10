# Released-model probability reliability

The study asks a product question: for English banking-intent routing, how many requests can a frozen rule accept, and what fraction of accepted requests are wrong? It separately measures whether stated probabilities match observed correctness.

Two studies share one protocol: the split seed, the 77-candidate request, raw temperature 1.0, the metrics and the policy-selection rule are identical, and both use the same 908 calibration / 896 policy / 1,276 test questions.

| Study | Models | Predictions | Results |
| --- | --- | --- | --- |
| Valen-Decision-1.5 | Valen-Decision-1.5-0.8B / 2B / 4B | new native inference, 2026-11-10 | `results/banking77-decision-1.5/` |
| Valen-Decision-1.0 | Valen-0.8B / 2B / 4B | retained JevalKit runs, 2026-10-16 | `results/banking77-v1/` |

BANKING77 contains 77 candidate intents per question, in English. The results describe text intent routing on this benchmark; image/video calibration, open-set rejection and other backends or business populations need their own measurement.

## Valen-Decision-1.5

### Measured results

All metrics use the same 1,276 held-out questions. Brier is a multiclass sum; ECE uses 15 equal-width bins. Lower is better for NLL/Brier/ECE. Calibration did not change any argmax.

| Model | Test accuracy | Temperature | NLL before → after | Brier before → after | ECE before → after |
| --- | ---: | ---: | ---: | ---: | ---: |
| Valen-Decision-1.5-0.8B | 82.29% | 0.891230 | 0.5926 → 0.5875 | 0.2502 → 0.2475 | 0.0387 → 0.0193 |
| Valen-Decision-1.5-2B | 86.91% | 0.968919 | 0.4133 → 0.4131 | 0.1794 → 0.1793 | 0.0205 → 0.0175 |
| Valen-Decision-1.5-4B | 89.89% | 1.072721 | 0.3551 → 0.3522 | 0.1491 → 0.1487 | 0.0166 → 0.0142 |

Temperature scaling lowers NLL, Brier and ECE for all three sizes. The raw probabilities are already close: raw ECE is 0.0387 / 0.0205 / 0.0166, and the fitted temperatures stay within 0.89–1.07. AURC moves by less than 0.0001 in either direction (0.8B 0.04022 → 0.04027, 2B 0.02318 → 0.02321, 4B 0.01559 → 0.01556).

| Model | Validation target | Test accepted / eligible | Test coverage | Errors / accepted | Test accepted error |
| --- | ---: | ---: | ---: | ---: | ---: |
| Valen-Decision-1.5-0.8B | 1% | 0 / 1276 | 0.00% | 0 / 0 | undefined (review all) |
| Valen-Decision-1.5-0.8B | 2% | 0 / 1276 | 0.00% | 0 / 0 | undefined (review all) |
| Valen-Decision-1.5-0.8B | 5% | 783 / 1276 | 61.36% | 27 / 783 | 3.45% |
| Valen-Decision-1.5-0.8B | 10% | 969 / 1276 | 75.94% | 68 / 969 | 7.02% |
| Valen-Decision-1.5-2B | 1% | 0 / 1276 | 0.00% | 0 / 0 | undefined (review all) |
| Valen-Decision-1.5-2B | 2% | 634 / 1276 | 49.69% | 5 / 634 | 0.79% |
| Valen-Decision-1.5-2B | 5% | 881 / 1276 | 69.04% | 20 / 881 | 2.27% |
| Valen-Decision-1.5-2B | 10% | 1111 / 1276 | 87.07% | 71 / 1111 | 6.39% |
| Valen-Decision-1.5-4B | 1% | 0 / 1276 | 0.00% | 0 / 0 | undefined (review all) |
| Valen-Decision-1.5-4B | 2% | 654 / 1276 | 51.25% | 3 / 654 | 0.46% |
| Valen-Decision-1.5-4B | 5% | 971 / 1276 | 76.10% | 18 / 971 | 1.85% |
| Valen-Decision-1.5-4B | 10% | 1233 / 1276 | 96.63% | 98 / 1233 | 7.95% |

No size has a qualifying rule at the 1% target, and 0.8B has none at 2%: those policies accept nothing and send every question to review. Every rule that accepts questions has an observed test error below its validation target.

### How the predictions were produced

Each model ran once on one NVIDIA H200 through JevalKit's native `ValenModel` adapter (`jevalkit.runner.run(model, "Banking77", …)`): all 3,080 requests, no limit, `execution=shared_state`, `attention=sdpa`, raw temperature 1.0, weights loaded with `dtype="auto"` (recorded parameter dtype `torch.bfloat16`), torch 2.10.0+cu128 and transformers 5.14.1. Every run finished with 3,080 valid predictions and no failures. `causal-conv1d` was not installed, so the linear-attention layers used the Transformers torch implementation.

`Banking77.tsv` was rebuilt from PolyAI commit `57ec275d` with JevalKit's original `prepare_banking77` (JevalKit commit `c1479e9`). Its SHA-256 is `f13205c4b54dbed549afcb86e26759b257aab96b763ceb8c0346c4d61c4d03a5`, the same file hash the Valen-Decision-1.0 reports record, and the test split has identical group hashes in both studies. The current `jeval run --data` choices do not list Banking77; the dataset loads through its local `Banking77.manifest.json`.

### Reproduce the replay

The root must contain `datasets/Banking77.tsv` and `results/full/<model>/Banking77/{manifest.json,predictions.jsonl,metrics.json}`. The release root is a copy of `Valen-Decision-1.5-main16k/` from [Valen-Decision-Index-Results](https://huggingface.co/datasets/Valen-Team/Valen-Decision-Index-Results) with `runs/<model>/DecisionIndex_Text/manifest.json`. `hub_identity.py` records each repository's commit and per-file SHA-256 (the Hub's LFS SHA-256 for weights and `tokenizer.json`, downloaded bytes for the other files); it is the only step that uses the network.

```bash
python -m evaluation.reliability.hub_identity --output output/hub-identity.json
python -m evaluation.reliability.banking77 --study decision-1.5 \
  --root /data/JevalKit --release-root /data/Valen-Decision-1.5-main16k \
  --hub-identity output/hub-identity.json \
  --output output/banking77-decision-1.5
python -m valen.calibrate audit \
  evaluation/reliability/results/banking77-decision-1.5/Valen-Decision-1.5-0.8B/report.json \
  --output output/audited-figures
```

The replay checks the dataset hash, the run's metrics-file checksums for predictions and manifest, and exact request coverage. It then requires every file hash the run recorded (weights, config, tokenizer, processor, chat template and runtime code: 18 files for 0.8B and 2B, 20 for 4B) to equal both the Hub repository at the recorded commit and the `artifacts_sha256` of the published Valen-Decision-1.5 Decision Index run manifest. The 1.5 release metadata has no `environment.json` and names revision `main`; the reports therefore record the Hub commits checked on 2026-11-10:

| Model | Hub commit |
| --- | --- |
| Valen-Decision-1.5-0.8B | `ae8bdf7606c0dbd627301174c9cb800240fb4ec7` |
| Valen-Decision-1.5-2B | `eb8da3dacc5e80f1e6f1cde28b6daea344fa3986` |
| Valen-Decision-1.5-4B | `0c67c26b8c5ac9d4bdb699ab31ecb4d4bf64f933` |

`results/banking77-decision-1.5/<model>/` holds the same files as the 1.0 study: aggregate report, temperature, protocol, four validation policies, CSV tables, SVG figures and `audit.json`, with `CHECKSUMS.sha256` and the `hub-identity.json` used by the replay at the top. The split, selection rule and source-corpus notes in the next section apply to both studies.

## Valen-Decision-1.0

This is a **retrospective replay** of retained JevalKit BANKING77 predictions dated 2026-10-16, with no new model inference. It is not a fresh blind test or production guarantee. BANKING77 contains 77 candidate intents per question. No result here establishes image/video calibration, open-set rejection, or transfer to another backend or business population.

### Reproduce without downloading models

The root must contain retained `datasets/Banking77.tsv` and `results/full/<model>/Banking77/{manifest.json,predictions.jsonl}`. The release root supplies `runs/<model>/environment.json` from the retained Valen Decision Index release metadata.

```bash
python -m evaluation.reliability.banking77 --study decision-1.0 \
  --root /data/JevalKit --release-root /data/Valen-Decision-Index-Results \
  --output output/banking77-reliability
```

The replay checks the dataset hash, original metrics-file checksums for predictions and manifest, exact request coverage, and every recorded weight/config/runtime-code hash against pinned release metadata. `--stdout-gzip` or the more compact `--stdout-xz` exports compressed aggregate reports to stdout and writes no remote files, useful when the logs are on a server and bandwidth is limited. It never exports prompts or per-question predictions.

A single source-independent seed `valen-reliability-banking77-v1` partitions normalized utterance groups into 30% calibration, 30% policy validation, 40% test using SHA-256 modulo 100. Exact case/whitespace-normalized utterances stay together. The resulting common cohort is 908 calibration questions, 896 policy questions, and 1,276 test questions. The seed and validation targets were frozen before computing these results.

Temperature minimizes calibration NLL; it is not chosen against test ECE. Validation selects the largest accepted set whose pointwise 95% Wilson upper error bound meets each target, with at least 30 accepted questions. Threshold search and distribution shift prevent this heuristic from being a formal risk guarantee. When no rule qualifies, its null threshold accepts nothing. Test results are measured after freezing all rules.

The source corpus lacks customer/conversation IDs; semantic near duplicates and training/pretraining/adaptation overlap cannot be ruled out. Public benchmark results were already available before this retrospective study. The 0.8B/2B runs used FP32 backbone parameters with BF16 backbone autocast; 4B used BF16 backbone parameters/autocast; the Mixer remained FP32. Historical tokenizer files were not individually hashed by the original JevalKit run. The identity binding covers recorded weights, config and runtime code; release metadata supplies the pinned revision. These limitations are included in every report.

### Inspect and reuse

`results/banking77-v1/<model>/` contains aggregate reports, temperatures, deployment recipe, four validation policies, CSV tables and standalone SVG figures. Source/data/prediction fingerprints and exact model revisions are recorded in the reports. No raw prompts or private evaluation logs are bundled. Full experiment replay requires access to the retained inputs; aggregate reports alone cannot recreate them.

```bash
python -m valen.calibrate audit \
  evaluation/reliability/results/banking77-v1/Valen-0.8B/report.json \
  --output output/audited-figures
```

This checks aggregate ECE, accuracy, NLL/Brier means, coverage/risk counts and AURC, then regenerates CSV/SVG artifacts. It verifies internal arithmetic, not the authenticity of model inputs. The underlying replay script and source hashes support the separate provenance audit.

Fit parameters for your own workflow using the [ticket-routing example](https://valen-team.github.io/cookbook/calibrated-routing/). Do not apply the historical 77-way artifact to unrelated candidate sets, prompts or backends. The [website evidence page](https://valen-team.github.io/docs/valen/reliability/) is generated from these reports, including regressions and accept-nothing policies.

Source dataset: [BANKING77, PolyAI](https://github.com/PolyAI-LDN/task-specific-datasets/tree/57ec275d8078af65b7731c2a98be812d844a6d6b/banking_data), CC-BY-4.0. The standard-library v1 protocol is mirrored in JevalKit's `jevalkit/reliability.py`; update and test both copies together.

### Measured results

All metrics use the same 1,276 held-out questions. Brier is a multiclass sum; ECE uses 15 equal-width bins. Lower is better for NLL/Brier/ECE. Calibration did not change any argmax.

| Model | Temperature | NLL before → after | Brier before → after | ECE before → after |
| --- | ---: | ---: | ---: | ---: |
| Valen-0.8B | 1.153418 | 1.3564 → 1.3240 | 0.5062 → 0.4996 | 0.0819 → 0.0518 |
| Valen-2B | 1.153104 | 1.1321 → 1.1053 | 0.4416 → 0.4368 | 0.0701 → 0.0337 |
| Valen-4B | 1.220072 | 1.2061 → 1.1814 | 0.4314 → 0.4413 | 0.0462 → 0.1006 |

4B improves NLL but worsens ECE on this test. Neither improvement nor automated coverage is assumed in advance.

| Model | Validation target | Test accepted / eligible | Test coverage | Errors / accepted | Test accepted error |
| --- | ---: | ---: | ---: | ---: | ---: |
| Valen-0.8B | 1% | 0 / 1276 | 0.00% | 0 / 0 | undefined (review all) |
| Valen-0.8B | 2% | 0 / 1276 | 0.00% | 0 / 0 | undefined (review all) |
| Valen-0.8B | 5% | 147 / 1276 | 11.52% | 1 / 147 | 0.68% |
| Valen-0.8B | 10% | 404 / 1276 | 31.66% | 40 / 404 | 9.90% |
| Valen-2B | 1% | 0 / 1276 | 0.00% | 0 / 0 | undefined (review all) |
| Valen-2B | 2% | 0 / 1276 | 0.00% | 0 / 0 | undefined (review all) |
| Valen-2B | 5% | 0 / 1276 | 0.00% | 0 / 0 | undefined (review all) |
| Valen-2B | 10% | 350 / 1276 | 27.43% | 19 / 350 | 5.43% |
| Valen-4B | 1% | 0 / 1276 | 0.00% | 0 / 0 | undefined (review all) |
| Valen-4B | 2% | 0 / 1276 | 0.00% | 0 / 0 | undefined (review all) |
| Valen-4B | 5% | 0 / 1276 | 0.00% | 0 / 0 | undefined (review all) |
| Valen-4B | 10% | 295 / 1276 | 23.12% | 20 / 295 | 6.78% |
