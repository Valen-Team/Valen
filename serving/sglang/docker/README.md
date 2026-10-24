# Deploy with the SGLang image

Run a local HTTP service with a Docker image containing the SGLang runtime and Valen plugin.

## Requirements

Linux x86-64, an NVIDIA GPU supporting FA3 (such as H100 or H200), a compatible CUDA driver, and Docker with Buildx and NVIDIA Container Toolkit. The image builds on the official SGLang runtime.

## Build the image

Get the [Valen source](https://github.com/Valen-Team/Valen) and build `valen-sglang:0.1.0-dev3` from the repository root. The image contains the serving runtime; model weights download on first startup.

```bash
docker buildx build --platform linux/amd64 --load \
  -t valen-sglang:0.1.0-dev3 \
  -f serving/sglang/docker/Dockerfile .
```

## Start the server

Start the container and wait for the model to load. The example mounts the Hugging Face cache on the host so later launches can reuse the weights.

| Setting | Example default | Purpose |
| --- | --- | --- |
| `--model-path` | `Valen-Team/Valen-2B` | Hub model name or complete checkpoint directory |
| `--served-model-name` | `valen` | Model name sent in requests |
| Port mapping | `127.0.0.1:30000:30000` | Access from the local host |

```bash
docker run --rm --gpus device=0 --ipc=host \
  -p 127.0.0.1:30000:30000 \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  valen-sglang:0.1.0-dev3 \
  --model-path Valen-Team/Valen-2B \
  --served-model-name valen \
  --host 0.0.0.0 --port 30000
```

## Send a request

Send a text request using SystemOne or Decisions:

| Protocol | Request structure | Predicate probability in the example |
| --- | --- | --- |
| SystemOne | `state` + questions keyed by ID | `answers.on_sofa.noul` |
| Decisions | `input` + question list | `answers[0].probability` |

The request `model` must match the configured served model name. `temperature` defaults to `1.0` and must be a finite number greater than zero.

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


## Send an image request

Run `python -m pip install requests` in the client environment and replace `cat.jpg` with a local PNG, JPEG, or WebP image. The Python example reads the file, infers its MIME type from the filename, and encodes it as a data URL.

SystemOne uses `image_url` content blocks in `state.messages`; Decisions uses `input_image` content blocks in `input`. Both examples send the model name `valen`.

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


## Supported scope

Full Valen Qwen3.5 Mixer checkpoints, `shared_state` execution, one GPU, TP=PP=1. Text and images support predicate, choice, and score questions: choice has 1–255 candidates; score has 2–255 ordered levels. For video, extract frames and submit them as image inputs.

See [SystemOne](https://valen-team.github.io/docs/valen/api/systemone/) and [Decisions](https://valen-team.github.io/docs/valen/api/decisions/) for question and answer structures.
