# Deploy with the SGLang plugin

Install the Valen plugin in an official SGLang environment and start a local HTTP inference service.

## Requirements

Linux x86-64, an NVIDIA GPU supporting FA3 (such as H100 or H200), a compatible CUDA driver, and Docker with NVIDIA Container Toolkit. You can also install the plugin directly in an existing SGLang Python environment.

## Build the kit

Get the [Valen source](https://github.com/Valen-Team/Valen) and build the wheel and kit from the repository root. Copy `serving-kit/sglang/` and a complete checkpoint to the GPU host.

```bash
python -m pip install build
python -m build --wheel serving/sglang
python serving/sglang/build_kit.py \
  serving/sglang/dist/valen_sglang-0.1.0.dev3-py3-none-any.whl \
  ./serving-kit/sglang
```

## Prepare the runtime

Use the official SGLang image. Replace the kit and checkpoint paths below:

```bash
docker run --rm -it --gpus device=0 --ipc=host \
  -p 127.0.0.1:30000:30000 \
  -v /absolute/path/to/serving-kit/sglang:/opt/valen-kit:ro \
  -v /absolute/path/to/Valen-2B:/models/Valen-2B:ro \
  --entrypoint bash \
  docker.io/lmsysorg/sglang:latest
```

Inside the container, `/opt/valen-kit` and `/models/Valen-2B` refer to those mounts. In an existing Python environment, substitute the actual paths below.

## Install and start

Install the plugin and start the service inside the container:

```bash
bash /opt/valen-kit/install.sh
MODEL=/models/Valen-2B MODEL_NAME=valen HOST=0.0.0.0 \
  CUDA_VISIBLE_DEVICES=0 bash /opt/valen-kit/serve.sh
```

| Setting | Example | Purpose |
| --- | --- | --- |
| `MODEL` | `/models/Valen-2B` | Complete checkpoint directory |
| `MODEL_NAME` | `valen` | Model name sent in requests |
| `HOST` / `PORT` | `0.0.0.0` / `30000` | Listen address and port inside the container |

After startup, access the service from the host at `http://127.0.0.1:30000`.

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

Command-line client:

```bash
python /opt/valen-kit/example_client.py --protocol decisions
python /opt/valen-kit/example_client.py --protocol systemone --image cat.jpg
```
