# GGUF 本地服务

这些 GGUF 需要 Valen 补丁版 llama.cpp/Ollama；官方未修改版本不能直接视为支持。首次在线体验见[版本选择](choosing_zh.md)。

## Linux x86-64 CPU 预编译包

运行包版本 **0.1.0**，目标 Ubuntu 22.04+、glibc 2.35+、AVX2/FMA/F16C/BMI2 CPU。验证范围与 SHA256 见[发布清单](https://github.com/Valen-Team/Valen/releases/tag/gguf-runtime-v0.1.0)。无需编译。先下载运行包和校验文件：

```sh
base=https://github.com/Valen-Team/Valen/releases/download/gguf-runtime-v0.1.0
curl -fLO "$base/valen-gguf-runtime-0.1.0-linux-x86_64.tar.gz"
curl -fLO "$base/SHA256SUMS"
sha256sum -c SHA256SUMS
tar -xzf valen-gguf-runtime-0.1.0-linux-x86_64.tar.gz
cd valen-gguf-runtime-0.1.0-linux-x86_64
```

优先使用已有模型文件。首次文本请求只需同尺寸的 Q8_0 文本文件，0.8B 约 0.761 GiB（磁盘大小，不是内存需求）。没有缓存时下载：

```sh
curl -fL https://huggingface.co/Valen-Team/Valen-0.8B-GGUF/resolve/main/Valen-0.8B-Q8_0.gguf -o Valen-0.8B-Q8_0.gguf
./serve -m "$PWD/Valen-0.8B-Q8_0.gguf" --load-mode none --device none -ngl 0 \
  --host 127.0.0.1 --port 8080 -c 4096 -b 4096 -ub 4096 \
  -t 4 --parallel 1 --no-warmup
```

另开终端发送文本请求：

```sh
curl --fail http://127.0.0.1:8080/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{"state":{"messages":[{"role":"user","content":"The film was excellent."}]},"questions":{"sentiment":{"type":"choice","instructions":"Classify the sentiment.","criteria":{"positive":"Positive","negative":"Negative"}}}}'
```

返回结构化候选概率，而不是生成文本。请求必须容纳于上下文和 batch 容量中。预编译包只声明新验证过的文本服务；图片与视频的既有源码验证不能代替此包的验证，视频未编入此包。其他尺寸替换模型文件，不能混用视觉模块。

源码 pin、补丁、许可证和源码构建/Ollama 说明由各模型仓库提供：

- [0.8B GGUF](https://huggingface.co/Valen-Team/Valen-0.8B-GGUF) · [runtime guide](https://huggingface.co/Valen-Team/Valen-0.8B-GGUF/blob/main/RUNTIME_GUIDE.md) · [custom Ollama](https://huggingface.co/Valen-Team/Valen-0.8B-GGUF/blob/main/OLLAMA_USAGE.md)
- [2B GGUF](https://huggingface.co/Valen-Team/Valen-2B-GGUF) · [runtime guide](https://huggingface.co/Valen-Team/Valen-2B-GGUF/blob/main/RUNTIME_GUIDE.md) · [custom Ollama](https://huggingface.co/Valen-Team/Valen-2B-GGUF/blob/main/OLLAMA_USAGE.md)
- [4B GGUF](https://huggingface.co/Valen-Team/Valen-4B-GGUF) · [runtime guide](https://huggingface.co/Valen-Team/Valen-4B-GGUF/blob/main/RUNTIME_GUIDE.md) · [custom Ollama](https://huggingface.co/Valen-Team/Valen-4B-GGUF/blob/main/OLLAMA_USAGE.md)


## 发布与验证记录

| Field | Value |
| --- | --- |
| Runtime | `0.1.0` |
| Artifact bytes | `31797146` |
| SHA256 | `2fcc903eb853b8c04498bb4ee716c7c3662d1d07eb5c2d89c11310716ae04c0e` |
| llama.cpp source base | `1a0b56cec73dac938e0602c974b21dc8e2862830` |
| Valen patch SHA256 | `20c8fcb50920ae8e96b158c30c277ab384fb9716d8d51fb01966b562b8e57732` |
| Compiler | GCC 11.4 / Ubuntu 22.04 |
| License | llama.cpp MIT + Valen patches Apache-2.0; notices and source included |

最终压缩包已在集群解压，逐文件校验并通过 0.8B Q8_0 真实文本请求；两次请求答案相同，解压后的结果也一致。没有重新下载模型。该包仅声明此文本验证范围；其他尺寸的既有源码验证见各模型卡。首次试用已有原生预编译包，不要求 Docker；本次不新增 Docker 镜像。
