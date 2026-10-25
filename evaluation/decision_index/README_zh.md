# Decision Index 公开集评测

[English](README.md) · **简体中文**

将已发布的 Valen 模型接入官方
[Decision Index 复现工具](https://github.com/apolinario/decision-index)。
适配器使用原生决策头和 shared-state 推理，直接复用上游 runner、返回值校验、冻结数据和评分器。

## 安装

在 Valen 仓库根目录执行：

```bash
python -m pip install -e ".[decision-index]"
python -m evaluation.decision_index --help
```

可选依赖固定到本次公开评测使用的上游提交
[`f5a6e4f`](https://github.com/apolinario/decision-index/tree/f5a6e4fd85fed05e77878dfc4966d0568bfb45ec)。
模块入口直接转发上游命令，并为 Python 3.10 补齐上游使用的 `hashlib.file_digest`。

## 准备冻结测试集

按照上游的
[重建与导入说明](https://github.com/apolinario/decision-index/tree/f5a6e4fd85fed05e77878dfc4966d0568bfb45ec#quickstart-local)
准备官方文件；重建数据还需要安装上游的 `rebuild` 可选依赖。
以下示例将测试集放在 `data/decision-index/suite-0.3`，也可以指定其他目录。
目录包含 `selected-rows.jsonl.gz`、`added-rows.jsonl.gz`、`gsm8k-rows.jsonl.gz`、
`excluded-questions.json` 和 `manifest.json`。

```bash
python -m evaluation.decision_index suite verify \
  --edition 0.3 --dir data/decision-index/suite-0.3
```

确认内容、排除清单和子集均匹配。评测流水线会在加载模型前校验冻结数据。
生成的数据和结果保留在 Git 忽略目录中。

## 运行评测

```bash
python -m evaluation.decision_index pipeline \
  --edition 0.3 \
  --suite-dir data/decision-index/suite-0.3 \
  --engine evaluation.decision_index.engine:ValenEngine \
  --model Valen-Team/Valen-2B \
  --option device=cuda \
  --option attention=sdpa \
  --out output/decision-index/0.3/Valen-2B \
  --compact
```

- `--model`：完整模型的本地目录或 Hugging Face 仓库名，支持
  `Valen-Team/Valen-0.8B`、`Valen-Team/Valen-2B` 和 `Valen-Team/Valen-4B`。
  Hub 模型下载到标准 Hugging Face 缓存。
- `--option revision=<commit>`：固定 Hub 模型版本，默认 `main`。
- `--option device=cpu` 或 `device=cuda:1`：选择运行设备。
- `--option attention=eager|sdpa|flash_attention_2`：选择其中一个 attention 后端。
  Flash Attention 需要兼容的安装包和 CUDA GPU，见
  [attention 配置](../../docs/configuration_zh.md)。

评测 CLEF 对应的 **0.2.1 公开集**时，将 `--edition` 改为 `0.2.1`，
`--suite-dir` 指向单独准备的该版冻结集，并使用独立输出目录。
该版需要基础数据、新增数据、排除清单和 manifest，不需要 `gsm8k-rows.jsonl.gz`。
两版分别使用各自的官方评分规则。

再次运行相同命令会按照上游机制续跑，并重试运行错误。
可添加 `--limit 10` 快速验证流程，此时的部分分数不能作为完整 benchmark 结果。

## ONNX CPU 评测

已发布的 ONNX 模型通过 `ValenONNXEngine` 接入同一套上游 runner 和评分器。使用 Python 3.11+，在 Valen 仓库根目录运行；模型推理由独立 SDK 执行，无需加载 PyTorch 模型。测试集仍使用上方已准备、验证的冻结文件。

```bash
python -m pip install \
  "valen-sdk[onnx] @ git+https://github.com/Valen-Team/valen-sdk-python.git" \
  "decision-index @ git+https://github.com/apolinario/decision-index.git@f5a6e4fd85fed05e77878dfc4966d0568bfb45ec"
python -m evaluation.decision_index pipeline \
  --edition 0.3 \
  --suite-dir data/decision-index/suite-0.3 \
  --engine evaluation.decision_index.onnx_engine:ValenONNXEngine \
  --model Valen-Team/Valen-0.8B-ONNX \
  --option variant=fp32 \
  --option threads=4 \
  --out output/decision-index/0.3/Valen-0.8B-ONNX-fp32 \
  --compact
```

`--model` 也可使用 2B、4B ONNX 仓库或本地模型目录。`--option variant=q8-mlp` 选择 CPU MLP 量化版本；`--option threads=4` 设置 CPU 线程数。已有完整缓存时，可加入 `--option offline=true` 禁止下载；`--option revision=<commit>` 固定模型版本。不同模型或变体应使用独立输出目录。

本入口使用原生决策头与 `shared_state`，保留上游问题与候选顺序，不截断候选。它是 CPU 评测适配器；应用内 CUDA、图片/视频与浏览器推理见 [ONNX 与 SDK](../../docs/onnx_zh.md)。


## 输入、精度与输出

保留全部问题 ID、选项及插入顺序。字符串保持原样；结构化 state、instructions 和
选项描述使用上游紧凑 UTF-8 JSON 格式；null 选项描述使用原选项名。
只有 state 和 questions 进入模型，标准答案始终留在评分侧。

继承模型已保存的决策头和最大长度，固定 shared-state 推理及 temperature=1.0。
保留发布参数的原始精度；CUDA 使用 BF16 主干 autocast，CPU 关闭 autocast。
实际参数与 autocast 精度记录在 `environment.json`。
超长输入和保留 tokenizer 控制符记为 `Unsupported`，不截断、不删选项；
其他异常保留为运行错误。

上游流水线输出 `results.jsonl`、`status.json`、`environment.json`、
`benchmark-summary.json`、`index.json` 和 `scores.json`。
`--compact` 使预测日志省略原始输入。请求计时包括编译、前向和原生答案转换，
排除模型加载和预热。

以上命令计算 **public index**。HF 的 Full score 还需要维护者运行私有评测；
AA Decision Index 和 HF Vision 榜是独立评测。官方速度由维护者在参考硬件上测量。
提交方式见上游的
[上榜说明](https://github.com/apolinario/decision-index#submitting-a-model-to-the-leaderboard)。
