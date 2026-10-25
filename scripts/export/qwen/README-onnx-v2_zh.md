# Valen 模块化 ONNX 导出

已发布模型的推理与 SDK 使用见 [ONNX 与 SDK 指南](../../../docs/onnx_zh.md)。本页用于从自己的原生检查点导出模型；直接推理无需导出环境。

从原生 merged Valen 检查点导出，保留已训练的视觉模块与 Mixer。复用固定版本 ORT GenAI 构建器，SDK 推理不依赖 Torch/Transformers。

默认命令为 `onnx_v2.py --source 原生目录 --output 新目录 --upstream 固定版本ORTGenAI目录`，输出须为空或不存在。支持 0.8B、2B、4B CPU FP32；CUDA 使用 `--provider cuda --precision fp16`。`--mlp-bits 8` 添加仅量化骨干 MLP 的候选，保护注意力、递推、归一化、视觉、嵌入与 Mixer；4 bit 仅作为实验候选。候选必须单独通过冻结质量门槛与实际设备测试。

`onnx_v2_web.py` 从 CPU bundle 和固定 SHA-256 的社区图模板构建浏览器版本，目前只验收 0.8B 架构。不读取社区学习权重，所有学习参数均映射自 Valen；图结构使用 Scan 作为浏览器回退，性能须独立测量。可用 `onnx_v2_quantize.py` 在现有 bundle 上添加 MLP 量化候选。

manifest 记录原生权重哈希、图输入输出 ABI、精度、依赖清单、文件哈希与导出器版本。导出完成不代表质量或设备验收通过。发布前应使用 SDK 的 `verify=True` 完成完整请求测试，并核对公开数据质量及实际算子放置。

源码与 SDK 单独维护；HF 权重仓库只放资产、manifest 和使用/验证文档。输出到新目录不修改任何已有模型或 HF 分支。详细环境、完整命令和模板来源见 [English](README-onnx-v2.md)。

浏览器模板只需下载固定 revision `8fe9dbe3fb39fe902e18df9be84b6a0ac77ae0f6` 的 `onnx/decoder_model_merged_fp16.onnx` 图文件，无需社区外部权重分片；英文说明提供了直接下载命令。
