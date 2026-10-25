# Valen modular ONNX export

For published models and runtime SDKs, use the [ONNX & SDK guide](../../../docs/onnx.md). This page covers exporting your own native checkpoint; inference does not need the export environment.

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

Bundles contain embedding, optional vision and fused decision graphs. Decision returns candidate logits after role pooling and Mixer. This graph layout avoids exporting language generation logits and avoids moving full hidden states back to the host. CPU/CUDA graphs use ORT contrib operators and are not universal standard-ONNX artifacts. Keep external data paths intact.

The manifest records native checkpoint/config hashes, graph ABI, precision, file hashes, exporter revision and component download lists. `quality_status` initially marks unvalidated candidates; export success never promotes a model automatically. Validate with `Valen.local(path, local_files_only=True, verify=True)` and complete native parity, frozen quality and target-device tests before publication.

Source export scripts and runtime code belong in their SDK/source repositories. HF weight repositories should contain model assets, manifests and usage/validation documents. Publication is a separate reviewed operation; exporting into a new directory does not modify any HF branch or existing model directory.
