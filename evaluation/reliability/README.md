# Released-model probability reliability

The first study asks a product question: for English banking-intent routing, how many requests can a frozen rule accept, and what fraction of accepted requests are wrong? It separately measures whether stated probabilities match observed correctness.

This is a **retrospective replay** of retained JevalKit BANKING77 predictions dated 2026-10-16, with no new model inference. It is not a fresh blind test or production guarantee. BANKING77 contains 77 candidate intents per question. No result here establishes image/video calibration, open-set rejection, or transfer to another backend or business population.

## Reproduce without downloading models

The root must contain retained `datasets/Banking77.tsv` and `results/full/<model>/Banking77/{manifest.json,predictions.jsonl}`. The release root supplies `runs/<model>/environment.json` from the retained Valen Decision Index release metadata.

```bash
python -m evaluation.reliability.banking77 \
  --root /data/JevalKit --release-root /data/Valen-Decision-Index-Results \
  --output output/banking77-reliability
```

The replay checks the dataset hash, original metrics-file checksums for predictions and manifest, exact request coverage, and every recorded weight/config/runtime-code hash against pinned release metadata. `--stdout-gzip` or the more compact `--stdout-xz` exports compressed aggregate reports to stdout and writes no remote files, useful when the logs are on a server and bandwidth is limited. It never exports prompts or per-question predictions.

A single source-independent seed `valen-reliability-banking77-v1` partitions normalized utterance groups into 30% calibration, 30% policy validation, 40% test using SHA-256 modulo 100. Exact case/whitespace-normalized utterances stay together. The resulting common cohort is 908 calibration questions, 896 policy questions, and 1,276 test questions. The seed and validation targets were frozen before computing these results.

Temperature minimizes calibration NLL; it is not chosen against test ECE. Validation selects the largest accepted set whose pointwise 95% Wilson upper error bound meets each target, with at least 30 accepted questions. Threshold search and distribution shift prevent this heuristic from being a formal risk guarantee. When no rule qualifies, its null threshold accepts nothing. Test results are measured after freezing all rules.

The source corpus lacks customer/conversation IDs; semantic near duplicates and training/pretraining/adaptation overlap cannot be ruled out. Public benchmark results were already available before this retrospective study. The 0.8B/2B runs used FP32 backbone parameters with BF16 backbone autocast; 4B used BF16 backbone parameters/autocast; the Mixer remained FP32. Historical tokenizer files were not individually hashed by the original JevalKit run. The identity binding covers recorded weights, config and runtime code; release metadata supplies the pinned revision. These limitations are included in every report.

## Inspect and reuse

`results/banking77-v1/<model>/` contains aggregate reports, temperatures, deployment recipe, four validation policies, CSV tables and standalone SVG figures. Source/data/prediction fingerprints and exact model revisions are recorded in the reports. No raw prompts or private evaluation logs are bundled. Full experiment replay requires access to the retained inputs; aggregate reports alone cannot recreate them.

```bash
python -m valen.calibrate audit \
  evaluation/reliability/results/banking77-v1/Valen-0.8B/report.json \
  --output output/audited-figures
```

This checks aggregate ECE, accuracy, NLL/Brier means, coverage/risk counts and AURC, then regenerates CSV/SVG artifacts. It verifies internal arithmetic, not the authenticity of model inputs. The underlying replay script and source hashes support the separate provenance audit.

Fit parameters for your own workflow using the [ticket-routing example](https://valen-team.github.io/cookbook/calibrated-routing/). Do not apply the historical 77-way artifact to unrelated candidate sets, prompts or backends. The [website evidence page](https://valen-team.github.io/docs/valen/reliability/) is generated from these reports, including regressions and accept-nothing policies.

Source dataset: [BANKING77, PolyAI](https://github.com/PolyAI-LDN/task-specific-datasets/tree/57ec275d8078af65b7731c2a98be812d844a6d6b/banking_data), CC-BY-4.0. The standard-library v1 protocol is mirrored in JevalKit's `jevalkit/reliability.py`; update and test both copies together.

## Measured results

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
