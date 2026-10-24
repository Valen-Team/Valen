# 使用 SGLang 镜像部署

使用包含 SGLang 运行时和 Valen 插件的 Docker 镜像，启动本地 HTTP 服务。

## 环境要求

Linux x86-64、支持 FA3 的 NVIDIA GPU（如 H100、H200）、兼容的 CUDA 驱动，以及安装 Buildx 和 NVIDIA Container Toolkit 的 Docker。镜像基于官方 SGLang 运行环境构建。

## 构建镜像

获取 [Valen 源码](https://github.com/Valen-Team/Valen)，在仓库根目录构建 `valen-sglang:0.1.0-dev3`。镜像包含服务运行环境，模型权重在首次启动时下载。

```bash
docker buildx build --platform linux/amd64 --load \
  -t valen-sglang:0.1.0-dev3 \
  -f serving/sglang/docker/Dockerfile .
```

## 启动服务

启动容器后等待模型加载完成。示例将 Hugging Face 缓存挂载到宿主机，后续启动可复用权重。

| 参数 | 默认示例 | 用途 |
| --- | --- | --- |
| `--model-path` | `Valen-Team/Valen-2B` | Hub 模型名或完整模型目录 |
| `--served-model-name` | `valen` | 请求中的模型名 |
| 端口映射 | `127.0.0.1:30000:30000` | 从本机访问服务 |

```bash
docker run --rm --gpus device=0 --ipc=host \
  -p 127.0.0.1:30000:30000 \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  valen-sglang:0.1.0-dev3 \
  --model-path Valen-Team/Valen-2B \
  --served-model-name valen \
  --host 0.0.0.0 --port 30000
```

## 发送请求

选择 SystemOne 或 Decisions 发送文本请求：

| 协议 | 请求结构 | 示例中的命题概率 |
| --- | --- | --- |
| SystemOne | `state` + 按 ID 索引的 `questions` | `answers.on_sofa.noul` |
| Decisions | `input` + 问题列表 | `answers[0].probability` |

请求中的 `model` 必须与启动时配置的模型名一致。`temperature` 默认为 `1.0`，必须是大于零的有限数值。

### Decisions

```bash
curl http://127.0.0.1:30000/v1/decisions \
  -H 'Content-Type: application/json' \
  -d '{
  "model": "valen",
  "input": "A cat is on the sofa.",
  "questions": [
    {
      "name": "on_sofa",
      "type": "predicate",
      "instructions": "The cat is on the sofa."
    }
  ],
  "temperature": 1
}'
```

### SystemOne

```bash
curl http://127.0.0.1:30000/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{
  "model": "valen",
  "state": "A cat is on the sofa.",
  "questions": {
    "on_sofa": {
      "type": "noul",
      "instructions": "The cat is on the sofa."
    }
  },
  "temperature": 1
}'
```


## 发送图片请求

在客户端运行 `python -m pip install requests`，将示例中的 `cat.jpg` 替换为本地 PNG、JPEG 或 WebP 图片。Python 示例会读取图片，按文件名识别 MIME 类型，并编码为 data URL。

SystemOne 使用 `state.messages` 中的 `image_url` 内容块；Decisions 使用 `input` 中的 `input_image` 内容块。请求中的模型名均为 `valen`。

### Decisions

```python
import base64
import mimetypes
from pathlib import Path

import requests

image_path = Path("cat.jpg")
mime_type = mimetypes.guess_type(image_path.name)[0]
image_b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
image_url = f"data:{mime_type};base64,{image_b64}"

response = requests.post(
    "http://127.0.0.1:30000/v1/decisions",
    json={
        "model": "valen",
        "input": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_image",
                        "image_url": image_url
                    },
                    {
                        "type": "input_text",
                        "text": "Inspect this image."
                    }
                ]
            }
        ],
        "questions": [
            {
                "name": "animal",
                "type": "choice",
                "instructions": "Which animal is visible?",
                "choices": [
                    {
                        "value": "cat",
                        "description": "A cat"
                    },
                    {
                        "value": "dog",
                        "description": "A dog"
                    }
                ]
            }
        ],
        "temperature": 1
    },
    timeout=120,
)
response.raise_for_status()
print(response.json()["answers"])
```

### SystemOne

```python
import base64
import mimetypes
from pathlib import Path

import requests

image_path = Path("cat.jpg")
mime_type = mimetypes.guess_type(image_path.name)[0]
image_b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
image_url = f"data:{mime_type};base64,{image_b64}"

response = requests.post(
    "http://127.0.0.1:30000/v1/systemone",
    json={
        "model": "valen",
        "state": {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_url
                            }
                        },
                        {
                            "type": "text",
                            "text": "Inspect this image."
                        }
                    ]
                }
            ]
        },
        "questions": {
            "animal": {
                "type": "choice",
                "instructions": "Which animal is visible?",
                "criteria": {
                    "cat": "A cat",
                    "dog": "A dog"
                }
            }
        },
        "temperature": 1
    },
    timeout=120,
)
response.raise_for_status()
print(response.json()["answers"])
```


## 支持范围

完整 Valen Qwen3.5 Mixer 检查点，`shared_state` 执行，单 GPU，TP=PP=1。支持文本和图片判断、选择与评分：Choice 为 1–255 个候选，Score 为 2–255 个有序等级。视频可先提取画面帧，再作为图片输入。

问题与答案结构见 [SystemOne](https://valen-team.github.io/zh-Hans/docs/valen/api/systemone/) 和 [Decisions](https://valen-team.github.io/zh-Hans/docs/valen/api/decisions/)。
