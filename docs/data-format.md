# Data format

**English** · [简体中文](data-format_zh.md)

Training and evaluation datasets use UTF-8 JSONL: one JSON record per line, with blank lines ignored. The reader loads the whole file into memory at once, and an empty file raises an error. The examples below are expanded for readability; collapse each record onto a single line when writing `.jsonl`. Ready-to-use files are in [data/smoke](../data/smoke/).

Inference accepts request objects directly: SystemOne uses `state` and a question mapping, while Decisions uses `input` and a question list. See [inference protocols](decisions.md), the [local inference tutorial](https://valen-team.github.io/cookbook/local-inference/) and [ONNX / SDKs](onnx.md). This page describes training and evaluation data files.

```json
{
  "group_id": "traffic-light-001",
  "request": {
    "state": "The traffic light is green.",
    "questions": {
      "color": {
        "type": "choice",
        "instructions": "Choose the observed traffic light color.",
        "criteria": {"red": "Red light", "green": "Green light", "unknown": "Not visible"}
      },
      "go": {
        "type": "noul",
        "instructions": "Only a green light permits crossing. Is crossing permitted?"
      },
      "risk": {
        "type": "score",
        "instructions": "Rate the risk of crossing under this light.",
        "criteria": ["Low: green light", "Medium: light not visible", "High: red light"]
      }
    }
  },
  "targets": {
    "color": {"probabilities": {"red": 0.0, "green": 1.0, "unknown": 0.0}},
    "go": {"probabilities": {"true": 1.0, "false": 0.0}},
    "risk": {"probabilities": {"0": 1.0, "1": 0.0, "2": 0.0}}
  },
  "assets": [],
  "meta": {"domain": "traffic", "modality": "text", "language_bucket": "en"}
}
```

## Records and questions

| Field | Convention | In the prompt? |
| --- | --- | --- |
| `group_id` | Required, non-empty group identifier, preferably a string; the same source material and its variants share one group ID | No |
| `request.state` | A text string, or a `{"messages": [...]}` object | Yes |
| `request.questions` | Non-empty mapping; its keys are the question IDs used in the returned answers | The ID is not; the question content is |
| `targets` | Optional mapping from question ID → label; must not contain unknown question IDs | No |
| `assets` | Optional media manifest; each entry has `path` and `sha256` | No |
| `meta` | Optional metadata; evaluation reads `record_id`, `domain`, `modality`, `language_bucket` | No |

Every question needs a lowercase `type` and a non-empty string `instructions`. The format of `criteria` depends on the task:

| `type` | `criteria` | Label keys | Order |
| --- | --- | --- | --- |
| `choice` | Mapping of 1–255 candidate names → non-empty descriptions | Candidate names; must cover all candidates | Training shuffles candidates together with their labels; inference keeps the input order |
| `noul` | Not needed; if provided, it is not used to build candidates | `"true"`, `"false"` | Fixed, true first |
| `score` | List of 2–255 non-empty grade descriptions | `"0"` to `"K-1"` | Keeps the list order, starting at 0 |

Choice candidate names are also part of the input text; question IDs are not. In the default `question` mode, each Score branch only sees its own grade description; in `shared_state` mode, all grade descriptions appear in one sequence and every grade is scored in a single forward pass. In both modes the output value is computed from list indices, in the range `[0, K-1]`. If your application needs a 1–5 or 0–100 scale, convert on the caller side.

## Labels and missing values

`probabilities` must cover exactly the full set of candidate keys for the question. Each value is a finite, non-negative number, and the sum is checked to equal 1 with `abs_tol=1e-6`. One candidate at 1 and the rest at 0 is a hard label; soft labels can be kept where a task calls for them.

Any of the following marks a question as unlabeled: omitting its target, setting the target to `null`, omitting `probabilities`, or setting `probabilities` to `null`. If there are no labels at all, omit `targets` or use `{}`; do not set the whole `targets` to `null`.

- Training and evaluation: unlabeled questions do not count toward the loss or metrics. In `question` mode they are skipped; in `shared_state` mode they stay in the input sequence as context. If a record has no labels at all, its media is never read.
- Inference: answers are returned for every question, labeled or not. Labels that are provided are still validated by the reader.
- Soft labels: count toward distribution metrics such as cross-entropy, Brier and RPS, but not toward hard-label accuracy or the Noul confusion matrix.

Extra label fields such as `label_source` and `votes` can record provenance; training does not read them. By default (`loss_reduction="state_mean"`), training first averages question losses within each state and then averages over states; `question_mean` averages over all labeled questions directly. Evaluation aggregates per question, so its denominator differs from that of default training.

## Images and video

Replace `state` with a message object:

```json
{
  "messages": [{
    "role": "user",
    "content": [
      {"type": "text", "text": "Determine the state of the traffic light in the image."},
      {"type": "image_url", "image_url": {"url": "assets/light.png"}}
    ]
  }]
}
```

Video entries use `{"type": "video_url", "video_url": {"url": "assets/clip.mp4"}}`. A message can hold multiple media entries. Supported roles are `user`, `assistant` and `system`; `content` can also be a plain text string.

Media paths are relative to **the directory containing the JSONL file**; local absolute paths also work. For example, `assets/light.png` in `data/train.jsonl` resolves to `data/assets/light.png`. Paths containing `://` are rejected, including HTTP URLs and `file://`; save the material locally first.

Compilation computes a SHA-256 digest for every media file actually used. Digests are compared only when `assets` has an entry with a matching path. The manifest can be omitted, and media that is not listed is still read and its digest recorded. The current code neither requires the manifest to cover all media nor checks for unused manifest entries. See [Compiler._messages / compile](../valen/data/compilers/qwen.py) for the logic.

`media_kwargs` is passed from the training config to the processor and then saved in the checkpoint for inference and evaluation. By default the base model's processor settings are used; there is currently no separate implementation of image resizing or video sampling. Media logs record the path, digest, vision grid and number of vision tokens, plus the frame indices, timestamps and other information returned by the processor.

## Splitting and checks

`group_id` exists to support splitting by source material, but the reader only checks that it is present. It does not generate train/validation/test splits, nor does it check for duplicates across files. When preparing real data, keep the same material, consecutive clips and paraphrased samples within one group, and check for media overlap across splits.

To check only the JSONL structure and labels, without loading the model or decoding media, run from the repository root:

```bash
python - <<'CHECK'
from valen.data.schema import read_jsonl
records = read_jsonl("data/smoke/train.jsonl")
print(f"{len(records)} records")
CHECK
```

Read errors include the file name and line number. Media existence, message roles, control tokens and length limits are checked later in `Compiler.compile`, so passing the check above does not mean the media is usable.

The synthetic set has 6 records and 15 questions, 13 of which are labeled. `text.jsonl` is the text-only subset of `train.jsonl`. See the [synthetic data notes](../data/smoke/README.md) for how it is generated and where the material comes from.
