# 数据格式

[English](data-format.md) · **简体中文**

训练与评估数据使用 UTF-8 JSONL：每行一条 JSON 记录，空行忽略。读取器一次将文件读入内存，空文件会报错。以下示例为方便阅读展开显示，写入 `.jsonl` 时需压成一行。可直接使用的文件见 [data/smoke](../data/smoke/)。

推理可直接传入请求对象；SystemOne 使用 `state` 与问题字典，Decisions 使用 `input` 与问题列表。示例见 [推理协议](decisions.md)、[Transformers Notebook](../cookbook/local_inference_zh.ipynb) 与 [ONNX / SDK](onnx_zh.md)。本页介绍训练与评测数据文件。

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

## 记录与问题

| 字段 | 约定 | 是否进入提示词 |
| --- | --- | --- |
| `group_id` | 必填非空组标识，建议使用字符串；同一素材及其变体使用相同组标识 | 否 |
| `request.state` | 文本字符串，或 `{"messages": [...]}` 对象 | 是 |
| `request.questions` | 非空映射；键是返回答案时使用的问题 ID | ID 不进入，问题内容进入 |
| `targets` | 可选的问题 ID → 标签映射；不能包含未知问题 ID | 否 |
| `assets` | 可选 media 清单，每项包含 `path`、`sha256` | 否 |
| `meta` | 可选元信息；评估读取 `record_id`、`domain`、`modality`、`language_bucket` | 否 |

每道题都需要小写的 `type` 和非空字符串 `instructions`。`criteria` 的格式由任务决定：

| `type` | `criteria` | 标签键 | 顺序 |
| --- | --- | --- | --- |
| `choice` | 1–255 个候选名 → 非空描述的映射 | 候选名，必须完整覆盖 | 训练打乱候选及对应标签；推理保留输入顺序 |
| `noul` | 不需要，提供时也不参与候选构造 | `"true"`、`"false"` | 固定 true 在前 |
| `score` | 2–255 个非空等级描述的列表 | `"0"` 到 `"K-1"` | 保留列表顺序，从 0 开始 |

Choice 的候选名也会作为输入文本，问题 ID 不会。默认的 `question` 模式下，Score 每个分支只看到自己的等级描述；`shared_state` 模式下，全部等级描述出现在同一序列中，一次前向即为全部等级评分。两种模式输出的数值都按列表索引计算，范围为 `[0, K-1]`。如果业务需要 1–5 分或 0–100 分，可以在调用端转换。

## 标签与缺失值

`probabilities` 必须恰好覆盖当前题的全部候选键。每个值是有限、非负的数字；总和按 `abs_tol=1e-6` 校验为 1。单个候选为 1、其余为 0 是硬标签；部分场景可以保留软标签。

以下写法都表示该题无标签：省略对应的 target、target 为 `null`、省略 `probabilities`、`probabilities` 为 `null`。若完全没有标签，省略 `targets` 或使用 `{}`，不要把整个 `targets` 写成 `null`。

- 训练、评估：无标签问题不计入 loss 和指标。`question` 模式下直接跳过这些问题；`shared_state` 模式下它们仍作为上下文保留在输入序列中。整条记录无标签时不读取其 media。
- 推理：返回所有问题的答案，即使没有标签。若提供了标签，读取器仍会校验它们。
- 软标签：参与交叉熵、Brier、RPS 等分布指标，不计入硬标签准确率和 Noul 混淆矩阵。

`label_source`、`votes` 等附加标签字段可以用于记录来源，训练不读取这些字段。训练默认（`loss_reduction="state_mean"`）先在每个 state 内平均题目损失，再按 state 平均；`question_mean` 则对全部有标签问题直接平均。评估按题汇总，因此与默认训练的分母不同。

## 图片和视频

将 `state` 换成消息对象即可：

```json
{
  "messages": [{
    "role": "user",
    "content": [
      {"type": "text", "text": "判断图片中的交通灯状态。"},
      {"type": "image_url", "image_url": {"url": "assets/light.png"}}
    ]
  }]
}
```

视频条目使用 `{"type": "video_url", "video_url": {"url": "assets/clip.mp4"}}`。可以在消息中放多个media条目。角色支持 `user`、`assistant`、`system`；`content` 也可以直接是文本字符串。

media路径相对于 **JSONL 文件所在目录**，也可使用本地绝对路径。例如 `data/train.jsonl` 中的 `assets/light.png` 对应 `data/assets/light.png`。包含 `://` 的路径会被拒绝，包括 HTTP URL 和 `file://`；需要先将素材保存到本地。

编译时会为实际使用的media计算 SHA-256。`assets` 中有路径匹配的条目时才比对摘要；清单可省略，未列出的media仍会被读取并记录摘要。当前代码不强制清单覆盖全部media，也不检查未使用的清单条目。相关逻辑见 [Compiler._messages / compile](../valen/data/compilers/qwen.py)。

`media_kwargs` 从训练配置传给处理器，随后保存在 checkpoint 中供推理和评估使用。默认读取基础模型的处理器设置；当前没有独立实现图片缩放或视频采样。media日志记录路径、摘要、视觉网格、视觉 token 数，以及处理器返回的帧索引、时间戳等信息。

## 划分与检查

`group_id` 用来支持按素材分组划分，但读取器只检查它存在，不会自动生成训练/验证/测试集，也不会检查跨文件重复。准备正式数据时，按组隔离同一素材、连续片段和改写样本，并检查跨划分的media重叠。

只检查 JSONL 结构和标签、不加载模型或解码media，可在仓库根目录运行：

```bash
python - <<'CHECK'
from valen.data.schema import read_jsonl
records = read_jsonl("data/smoke/train.jsonl")
print(f"{len(records)} records")
CHECK
```

读取错误会带文件名和行号。media存在性、消息角色、控制 token 和长度限制在之后的 `Compiler.compile` 中检查，因此通过上述校验不代表media已可用。

合成集包含 6 条记录、15 道题，其中 13 道有标签。`text.jsonl` 是 `train.jsonl` 的纯文本子集。生成方法及素材说明见[合成数据说明](../data/smoke/README.md)。
