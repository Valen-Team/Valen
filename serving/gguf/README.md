# Valen GGUF runtime

The Valen GGUF runtime is a patched llama.cpp `llama-server` that answers Valen decision requests with candidate probabilities and boxes. User instructions: [English](https://valen-team.github.io/docs/valen/gguf/) · [中文](https://valen-team.github.io/zh-Hans/docs/valen/gguf/).

| Runtime | Release | Models |
| --- | --- | --- |
| 0.2.0 | [`gguf-runtime-v0.2.0`](https://github.com/Valen-Team/Valen/releases/tag/gguf-runtime-v0.2.0) | Valen-Decision-1.5 (text, image, bbox) and Valen-Decision-1.0 |
| 0.1.0 | [`gguf-runtime-v0.1.0`](https://github.com/Valen-Team/Valen/releases/tag/gguf-runtime-v0.1.0) | Valen-Decision-1.0 |

## `valen-decision-1.5-runtime.patch`

The source change from runtime 0.1.0 to 0.2.0: Valen-Decision-1.5 block attention, noul-rubric folding and `bbox` questions on image requests through `/v1/systemone` and `/v1/decisions`, plus the converter support for the new weights. It touches 17 files; SHA256 `857d670488e0c796b4346b893353cc0d7a5b588cf0d5393c88d58d939f843bb8`.

**Base.** The patch applies to the runtime 0.1.0 source snapshot, `source/llama-source.tar.gz` inside `valen-gguf-runtime-0.1.0-linux-x86_64.tar.gz` (snapshot SHA256 `f6942f3c6638e57c7fc38319af60b391393546f12b505eb0c8401bb7afa29149`; llama.cpp base `1a0b56cec73dac938e0602c974b21dc8e2862830` with the 0.1.0 Valen patch already applied).

**Apply and build.**

```sh
base=https://github.com/Valen-Team/Valen/releases/download/gguf-runtime-v0.1.0
curl -fLO "$base/valen-gguf-runtime-0.1.0-linux-x86_64.tar.gz"
tar -xzf valen-gguf-runtime-0.1.0-linux-x86_64.tar.gz
tar -xzf valen-gguf-runtime-0.1.0-linux-x86_64/source/llama-source.tar.gz
cd llama.cpp
patch -p1 < /path/to/Valen/serving/gguf/valen-decision-1.5-runtime.patch
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON -DGGML_BMI2=ON \
  -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DGGML_CCACHE=OFF
cmake --build build --target llama-server llama-quantize --parallel 8
```

The result is the tree shipped as `source/llama-source.tar.gz` in the 0.2.0 package, which is ready to build as is.

**Convert.** From the patched tree, with a native `Valen-Decision-1.5-<size>` directory as `$SOURCE`:

```sh
python convert_hf_to_gguf.py "$SOURCE" --outtype f32 --outfile Valen-Decision-1.5-<size>-F32.gguf
python convert_hf_to_gguf.py "$SOURCE" --outtype f32 --mmproj --outfile vision-native-limits-F32.gguf
build/bin/llama-quantize Valen-Decision-1.5-<size>-F32.gguf Valen-Decision-1.5-<size>-Q8_0.gguf Q8_0 8
```

The published files, their hashes and the measured differences against the native model are in `MANIFEST.json` of each `Valen-Team/Valen-Decision-1.5-<size>-GGUF` repository. `docs/gguf-runtime-manifest.json` records the 0.2.0 release artifact.
