# Valen 合成训练测试数据

由 `scripts/data/make_smoke_data.py` 生成，素材与规则均为本项目合成内容。

样本已随仓库提供。需要重新生成时，先激活项目的 Python 环境，再在仓库根目录运行 `python scripts/data/make_smoke_data.py`；该命令会覆盖本目录中的合成数据和媒体。

| group_id | 输入 | 有标签问题 |
| --- | --- | --- |
| synthetic-text-0 | 中文红灯描述 | Choice、Noul、Score |
| synthetic-text-1 | 英文绿灯描述 | Choice、Noul、Score |
| synthetic-image | 绿灯 PNG | Choice、Noul、Score |
| synthetic-video | 红转绿 MP4，判断最后状态 | Choice、Noul、Score |
| synthetic-soft | 三位标注者的风险投票 | Score 软标签；另有缺标签 Noul |
| synthetic-unlabeled | 无可见交通灯的文字 | 缺标签 Noul，训练时整条跳过 |

`train.jsonl` 共 6 个 state、15 个问题，其中 13 个有标签。`text.jsonl` 为纯文本阶段使用的子集，包含 4 个 state、9 个问题，其中 7 个有标签。两者不能作为独立的训练/验证划分。测试数据仅检查代码路径，不用于评估模型效果。

推理处理 `train.jsonl` 时返回 6 行响应，包含全部 15 道题；评估生成 13 行逐题预测，`metrics.json` 中记录 `records=6`、`questions=13`。这些计数可用来检查命令是否使用了预期数据。字段约定见[数据格式](https://valen-team.github.io/zh-Hans/docs/valen/data-format/)，输出区别见[推理与评估](https://valen-team.github.io/zh-Hans/docs/valen/evaluation/)。

媒体文件路径相对于该目录；`assets` 中记录 SHA256。图像为 128×128；视频为 4 fps、16 帧、4 秒，前半段红灯、后半段绿灯。实际输入帧由固定版本官方处理器采样，训练日志中另存帧索引与时间戳。
