# Changelog

All notable changes to the Valen repository are recorded here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

Minor versions mark model releases; patch versions mark code or documentation changes without new weights. Each release lists the Hugging Face checkpoints it was validated against, so downstream integrations can pin a code version to a model revision.

## [Unreleased]

### Code
- Add probability calibration, routing policies and released-model evidence.
- Remove the `release-evidence` workflow, `validate_release_report.py` and `validation/release-gate.json`; the release checklist in `scripts/validation/README.md` now stands on its own.

### Docs
- Add `CONTRIBUTING.md` and `SECURITY.md`.
- Add this changelog, a releasing guide in `CONTRIBUTING.md`, and the 0.3.0 version in `pyproject.toml` and the citation. Trim README News to the three latest entries.
- Add community projects and the Jev-ToM community evaluation.
- Highlight handling of unknown choices; link API data handling documentation.

## [0.3.0] - 2026-10-26

### Models
- Qualified F32 and Q8_0 GGUFs for Valen-0.8B, Valen-2B and Valen-4B, served by the Valen-patched llama.cpp and Ollama builds (`gguf-runtime-v0.1.0`), which return candidate probabilities instead of generated text. The prebuilt package covers text and image input; video is available from a source build.

### Code
- Add the verified GGUF CPU runtime and pin the verified model download.
- Add fixed backend regression checks.

### Docs
- Unified bilingual version guidance; CPU runtime system library notes.
- Record PyPI `valen-sdk==0.2.0` and npm `@valen-team/sdk@0.2.0` publication.
- Record Valen-0.8B training speed comparisons.
- Move user guides to the website; retire migrated notebooks after live Cookbook verification.

## [0.2.3] - 2026-10-25

### Models
- Updated modular ONNX exports for Valen-0.8B, Valen-2B and Valen-4B.

### Code
- Add dual inference protocols and simplify serving deployment.
- Add modular ONNX export tooling.

### Docs
- SDK guides for the Python SDK, the JavaScript / TypeScript SDK and the C++ text integration.

## [0.2.2] - 2026-10-22

### Code
- Public Decision Index adapter and reproduction guides (2026-10-16).
- Add a validated SGLang plugin serving kit.
- Add a validated vLLM serving plugin and a pinned Valen vLLM container for GPU validation.

### Docs
- Bilingual SGLang container inference guide; vLLM image deployment guides and startup commands in both READMEs.
- 2026-10-21: web chat, bilingual documentation and the free key-less decision API went live.

## [0.2.1] - 2026-10-13

### Models
- ONNX and GGUF versions of the Valen models published on Hugging Face.
- API access to `valen-preview-0923`, `valen-0-8b`, `valen-2b` and `valen-4b`.

### Code
- Portable ONNX and GGUF export tools with validation.
- Support fine-tuning from released Valen checkpoints.

### Docs
- Bilingual cookbook with API and local inference examples; API cookbooks default to Valen-2B.
- English docs alongside the Chinese originals; repository citation.

## [0.2.0] - 2026-10-07

### Models
- Valen-0.8B, Valen-2B and Valen-4B released with text, image and video support.
- VisualDecisionBench benchmark released.

### Code
- Qwen shared-state training and portable model exports.
- Full Qwen SFT support; portable v1 full SFT training configs archived.
- VisualDecisionBench evaluation.

### Docs
- Refreshed bilingual quick start; dataset tables replaced with concise descriptions.
- Previous Preview READMEs archived under `history/`.

## [0.1.1] - 2026-09-30

### Code
- Flash Attention 2 support, multi-node multi-GPU training and improved GPU memory utilization.
- Dual-encoder decisions with refactored architecture backends.
- Caption alignment pretraining and decision SFT experiments.
- Qwen decision heads and portable cluster configuration.
- JevBench evaluation and native model baselines.

## [0.1.0] - 2026-09-23

### Models
- Valen-preview-0923 released with its training data.

### Code
- Initial open-source release: model implementation, data processing, SFT and experimental RLCD training, inference and evaluation commands.

[Unreleased]: https://github.com/Valen-Team/Valen/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/Valen-Team/Valen/compare/v0.2.3...v0.3.0
[0.2.3]: https://github.com/Valen-Team/Valen/compare/v0.2.2...v0.2.3
[0.2.2]: https://github.com/Valen-Team/Valen/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/Valen-Team/Valen/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/Valen-Team/Valen/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/Valen-Team/Valen/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/Valen-Team/Valen/releases/tag/v0.1.0
