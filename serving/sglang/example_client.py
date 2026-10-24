"""Text and optional image example for SystemOne and Decisions endpoints."""

import argparse
import base64
import json
import mimetypes
from pathlib import Path

import requests

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--base-url", default="http://127.0.0.1:30000")
parser.add_argument("--model", default="valen")
parser.add_argument("--image", type=Path)
parser.add_argument(
    "--protocol", choices=["systemone", "decisions"], default="decisions"
)
args = parser.parse_args()
state = "A cat is sitting on the sofa."
if args.image:
    mime = mimetypes.guess_type(args.image.name)[0]
    if not mime or not mime.startswith("image/"):
        parser.error("--image must be a recognized image file")
    data = base64.b64encode(args.image.read_bytes()).decode("ascii")
    state = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_image",
                        "image_url": f"data:{mime};base64,{data}",
                    },
                    {"type": "input_text", "text": "Inspect this image."},
                ],
            }
        ]
    }
payload = {
    "model": args.model,
    "input": state["messages"] if isinstance(state, dict) else state,
    "questions": [
        {
            "name": "animal",
            "type": "choice",
            "instructions": "Which animal is present?",
            "choices": [
                {"value": "cat", "description": "A cat"},
                {"value": "dog", "description": "A dog"},
            ],
        },
        {
            "name": "on_sofa",
            "type": "predicate",
            "instructions": "The cat is on the sofa.",
        },
        {
            "name": "clarity",
            "type": "score",
            "instructions": "How clearly is the animal identified?",
            "levels": [
                {"label": text, "description": text}
                for text in ["Unclear", "Partially clear", "Clear"]
            ],
        },
    ],
}
if args.protocol == "systemone":
    internal_state = payload.pop("input")
    if isinstance(internal_state, list):
        for message in internal_state:
            for part in message["content"]:
                if part["type"] == "input_image":
                    part.update(type="image_url", image_url={"url": part["image_url"]})
                else:
                    part["type"] = "text"
        internal_state = {"messages": internal_state}
    payload["state"] = internal_state
    questions = {}
    for q in payload["questions"]:
        item = {
            "type": "noul" if q["type"] == "predicate" else q["type"],
            "instructions": q["instructions"],
        }
        if q["type"] == "choice":
            item["criteria"] = {c["value"]: c["description"] for c in q["choices"]}
        elif q["type"] == "score":
            item["criteria"] = [level["description"] for level in q["levels"]]
        questions[q["name"]] = item
    payload["questions"] = questions
response = requests.post(
    args.base_url.rstrip("/") + "/v1/" + args.protocol,
    json=payload,
    timeout=120,
)
response.raise_for_status()
print(json.dumps(response.json(), ensure_ascii=False, indent=2))
