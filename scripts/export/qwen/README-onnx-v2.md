# Valen modular ONNX export

For published models and runtime SDKs, use the [ONNX & SDK guide](https://valen-team.github.io/docs/valen/onnx/). This page covers exporting your own native checkpoint; inference does not need the export environment.

Export the native merged **Valen** checkpoint, including its trained vision tower and Mixer. The pipeline reuses Microsoft's pinned Qwen3.5 ORT GenAI builder and Valen's component exporter. Runtime SDKs are separate packages and do not need Torch/Transformers.

Use a separate export environment:

```sh
python -m venv export-env
. export-env/bin/activate
pip install -r scripts/export/qwen/requirements-onnx-v2-export.txt
git clone https://github.com/microsoft/onnxruntime-genai.git /path/to/onnxruntime-genai
git -C /path/to/onnxruntime-genai checkout 6b7ff3ca3efa15da219050fdb4ecf8bed67875af
python scripts/export/qwen/onnx_v2.py \
  --source /path/to/Valen-0.8B --output /path/to/new-cpu-bundle \
  --upstream /path/to/onnxruntime-genai
```

Run from the Valen repository root. Output must be new or empty. Repeat with native 2B/4B directories. The exporter checks the upstream revision and creates vision/head components.

CPU FP32 is the default. `--provider cuda --precision fp16` exports the CUDA backbone/embedding variant, preserving FP32 vision, pooling and Mixer. Provider and precision are validated as a pair. For a protected CPU quantization candidate, add `--mlp-bits 8`; this quantizes only backbone MLP MatMul weights and leaves attention/recurrent, normalization, vision, embeddings and Mixer at their original precision. `--mlp-bits 4` is an experimental candidate. The quantizer explicitly excludes every non-selected node because upstream `nodes_to_include` alone is additive. Every candidate still needs the frozen quality gate and a target-device runtime test.

```sh
# Add a quantized candidate to an existing CPU base bundle:
python scripts/export/qwen/onnx_v2_quantize.py \
  --model /path/to/cpu-bundle --base-variant fp32 --bits 8
```

The pinned 0.8B browser graph is built separately:

```sh
python scripts/export/qwen/onnx_v2_web.py \
  --source /path/to/cpu-bundle --output /path/to/new-browser-bundle \
  --template /path/to/community-fp16.onnx
python scripts/export/qwen/onnx_v2_quantize.py \
  --model /path/to/new-browser-bundle --base-variant web-portable --bits 4
```

Download only `onnx/decoder_model_merged_fp16.onnx` from the pinned community revision as the graph template; its external learned weight shards are not required:

```sh
hf download onnx-community/Qwen3.5-0.8B-ONNX onnx/decoder_model_merged_fp16.onnx \
  --revision 8fe9dbe3fb39fe902e18df9be84b6a0ac77ae0f6 --local-dir /path/to/template
```

Pass `/path/to/template/onnx/decoder_model_merged_fp16.onnx` to `--template`.

The browser builder accepts only the verified Valen 0.8B architecture and template SHA-256 `c2f332890fb1cdb8ee05e21a52e871ba3333f6149a30de97c4cfb0227207e674` from `onnx-community/Qwen3.5-0.8B-ONNX` revision `8fe9dbe3fb39fe902e18df9be84b6a0ac77ae0f6`. No community learned weight shard is read. All learned initializers are mapped to Valen tensors, and a tensor mapping report is written. Standard Scan control flow replaces unsupported GatedDeltaNet in this graph; browser execution and performance must be measured independently.

Valen-Decision-1.5 checkpoints export the same layout with two additions. The decision graph takes `block_mask`, an additive float32 `[sequence, sequence]` mask for the full-attention layers that the SDKs build from the compiled prefix length and segment ids, and returns `final_hidden` next to `logits`. A `grounding` graph answers bbox questions from the decision and image-token rows of `final_hidden`.

This holds for every variant. `onnx_v2_attention.py` replaces each `GroupQueryAttention` node with explicit attention over `block_mask` in the CPU FP32 graph, the CUDA FP16 graph and the browser graph built from the community template. `block_mask` stays float32 at the graph boundary; FP16 graphs clip it to the float16 range and cast it inside the graph, so the SDKs feed the same mask to all variants. `final_hidden` is float32 in all variants.

Add a validated variant to a bundle that already holds others with `onnx_v2_assemble.py`:

```sh
python scripts/export/qwen/onnx_v2_assemble.py \
  --base /path/to/cpu-bundle --export /path/to/new-cuda-bundle \
  --variant cuda-fp16 --output /path/to/staging-bundle
python scripts/export/qwen/onnx_v2_assemble.py \
  --base /path/to/cpu-bundle --export /path/to/new-browser-bundle \
  --variant web-portable --output /path/to/staging-bundle
```

Files whose content the base bundle already holds keep their name and are shared between variants (vision, grounding, head weights, tokenizer files; for `web-portable` also the embedding). The remaining files are written as `onnx/<variant>-<name>`, with the external-data locations inside the graphs rewritten, and the merged `valen_onnx.json` gains the variant's `graphs`, `components`, `files`, `sha256`, `io_schema` and `download_bytes`. The base directory is only read; the staging directory holds the new variant's files and the merged manifest, ready to validate with `Valen.local(staging, variant=..., verify=True)` and to upload next to the published files.

Bundles contain embedding, optional vision and fused decision graphs. Decision returns candidate logits after role pooling and Mixer. This graph layout avoids exporting language generation logits and avoids moving full hidden states back to the host. CPU/CUDA graphs use ORT contrib operators and are not universal standard-ONNX artifacts. Keep external data paths intact.

The manifest records native checkpoint/config hashes, graph ABI, precision, file hashes, exporter revision and component download lists. `quality_status` initially marks unvalidated candidates; export success never promotes a model automatically. Validate with `Valen.local(path, local_files_only=True, verify=True)` and complete native parity, frozen quality and target-device tests before publication.

Source export scripts and runtime code belong in their SDK/source repositories. HF weight repositories should contain model assets, manifests and usage/validation documents. Publication is a separate reviewed operation; exporting into a new directory does not modify any HF branch or existing model directory.
