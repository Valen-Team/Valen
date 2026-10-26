# GGUF local service

These models require the Valen-patched llama.cpp/Ollama. Support in official unmodified releases is not claimed. For the quickest online trial, see [version selection](choosing.md).

## Linux x86-64 CPU prebuilt package

Runtime version **0.1.0** targets Ubuntu 22.04+, glibc 2.35+ and AVX2/FMA/F16C/BMI2 CPUs. See the [release manifest](https://github.com/Valen-Team/Valen/releases/tag/gguf-runtime-v0.1.0) for validation scope and SHA256. No compilation is required. Download the runtime and checksum file:

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
curl -fL https://huggingface.co/Valen-Team/Valen-0.8B-GGUF/resolve/main/Valen-0.8B-Q8_0.gguf -o Valen-0.8B-Q8_0.gguf
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

The response contains structured candidate probabilities, not generated text. The whole request must fit the context and batch capacities. This package claims newly verified text serving only; existing source-build image/video tests do not qualify this package, and video is disabled in this build. Substitute another size’s model file as needed; never mix vision modules between sizes.

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

The final archive was extracted on the cluster, checked file by file, and passed a real 0.8B Q8_0 text request. Two requests returned identical answers, including the extracted-package run. No model was downloaded again. This package claims this text scope only; the model cards record prior source-build validation for other sizes. The native prebuilt package provides the first-run path without Docker; no additional Docker image is required for this release.
