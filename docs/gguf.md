# GGUF local service

These models require the Valen-patched llama.cpp/Ollama. Support in official unmodified releases is not claimed. For the quickest online trial, see [version selection](choosing.md).

## Linux x86-64 CPU prebuilt package

Runtime version **0.1.0** targets Ubuntu 22.04+, glibc 2.35+ and AVX2/FMA/F16C/BMI2 CPUs. See the [release manifest](https://github.com/Valen-Team/Valen/releases/tag/gguf-runtime-v0.1.0) for validation scope and SHA256. No compilation is required. Download the runtime and checksum file:

System prerequisites are `curl`, `tar`, `sha256sum`, `libstdc++6` and `libgomp1`. On a minimal Ubuntu installation, install missing components first:

```sh
sudo apt-get update
sudo apt-get install -y curl ca-certificates coreutils tar libstdc++6 libgomp1
```

```sh
base=https://github.com/Valen-Team/Valen/releases/download/gguf-runtime-v0.1.0
curl -fLO "$base/valen-gguf-runtime-0.1.0-linux-x86_64.tar.gz"
curl -fLO "$base/SHA256SUMS"
sha256sum -c SHA256SUMS
tar -xzf valen-gguf-runtime-0.1.0-linux-x86_64.tar.gz
cd valen-gguf-runtime-0.1.0-linux-x86_64
```

Reuse existing model files first. A text request needs only the matching Q8_0 text file; 0.8B is about 0.761 GiB on disk, not a RAM requirement. If uncached, download:

```sh
curl -fL https://huggingface.co/Valen-Team/Valen-0.8B-GGUF/resolve/e24c6ad38a83d300ae7fc83917a1a54a0e416455/Valen-0.8B-Q8_0.gguf -o Valen-0.8B-Q8_0.gguf
echo '930d29c39c7104bb69a028a0605fad5eba0949e1f3ad4a3a3bbca13716c4b9d6  Valen-0.8B-Q8_0.gguf' | sha256sum -c -
./serve -m "$PWD/Valen-0.8B-Q8_0.gguf" --load-mode none --device none -ngl 0 \
  --host 127.0.0.1 --port 8080 -c 4096 -b 4096 -ub 4096 \
  -t 4 --parallel 1 --no-warmup
```

In another terminal, send a text request:

```sh
curl --fail http://127.0.0.1:8080/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{"state":{"messages":[{"role":"user","content":"The film was excellent."}]},"questions":{"sentiment":{"type":"choice","instructions":"Classify the sentiment.","criteria":{"positive":"Positive","negative":"Negative"}}}}'
```

The response contains structured candidate probabilities, not generated text. The whole request must fit the context and batch capacities. The original release qualified text serving only. P1 additionally verified this exact prebuilt binary on cached 0.8B Q8_0 fixed-image cases; see the [recorded scope and preserved failures](https://valen-team.github.io/cookbook/validation/). Direct video is not compiled into this package; source-build video tests do not qualify it. Substitute another size’s model file as needed; never mix vision modules between sizes.

Source pins, patches, licenses and source-build/Ollama instructions are available in each model repository:

- [0.8B GGUF](https://huggingface.co/Valen-Team/Valen-0.8B-GGUF) · [runtime guide](https://huggingface.co/Valen-Team/Valen-0.8B-GGUF/blob/main/RUNTIME_GUIDE.md) · [custom Ollama](https://huggingface.co/Valen-Team/Valen-0.8B-GGUF/blob/main/OLLAMA_USAGE.md)
- [2B GGUF](https://huggingface.co/Valen-Team/Valen-2B-GGUF) · [runtime guide](https://huggingface.co/Valen-Team/Valen-2B-GGUF/blob/main/RUNTIME_GUIDE.md) · [custom Ollama](https://huggingface.co/Valen-Team/Valen-2B-GGUF/blob/main/OLLAMA_USAGE.md)
- [4B GGUF](https://huggingface.co/Valen-Team/Valen-4B-GGUF) · [runtime guide](https://huggingface.co/Valen-Team/Valen-4B-GGUF/blob/main/RUNTIME_GUIDE.md) · [custom Ollama](https://huggingface.co/Valen-Team/Valen-4B-GGUF/blob/main/OLLAMA_USAGE.md)


## Release and validation record

| Field | Value |
| --- | --- |
| Runtime | `0.1.0` |
| Artifact bytes | `31797146` |
| SHA256 | `2fcc903eb853b8c04498bb4ee716c7c3662d1d07eb5c2d89c11310716ae04c0e` |
| llama.cpp source base | `1a0b56cec73dac938e0602c974b21dc8e2862830` |
| Valen patch SHA256 | `20c8fcb50920ae8e96b158c30c277ab384fb9716d8d51fb01966b562b8e57732` |
| Compiler | GCC 11.4 / Ubuntu 22.04 |
| License | llama.cpp MIT + Valen patches Apache-2.0; notices and source included |

The final archive was extracted on the cluster, checked file by file, and passed a real 0.8B Q8_0 text request. Two requests returned identical answers, including the extracted-package run. No model was downloaded again. That original text check remains unchanged; additional P1 fixed-image evidence is linked above. The model cards record prior source-build validation for other sizes. The native prebuilt package provides the first-run path without Docker; no additional Docker image is required for this release.

## P1 image qualification

P1 reused this exact extracted runtime, with cached 0.8B Q8_0 and the matching published `vision-native-limits-F32.gguf`. Add `--mmproj /your/cache/vision-native-limits-F32.gguf --no-mmproj-offload --image-min-tokens 64 --image-max-tokens 256` to the launch command above; keep model/vision sizes matched. The image module SHA256 is `db28fe68408846202ad9dd8ac71c4f43df7d8665e0d94422fa1e91e3029fe03b` for the tested 0.8B file. Reuse caches and verify server metadata before downloading.

[Cookbook](https://valen-team.github.io/cookbook/) has executable text/image applications and sampled-frame video tagging. Those image requests passed on the prebuilt binary; **direct video remains uncompiled**. [P1 validation](https://valen-team.github.io/cookbook/validation/) records exact versions, input budgets, errors and CPU/RSS measurements. No new runtime archive or tag replaces 0.1.0.
