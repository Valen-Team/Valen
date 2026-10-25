"""Standalone Qwen + decision head. / 可独立加载的 Qwen 和决策头。"""
import torch
from transformers import AutoProcessor
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model, Qwen3_5PreTrainedModel

from .configuration_valen import ValenQwenConfig
from .heads import build_head
from .runtime_model import ValenQwen
from .compiler import Compiler
from .responses import answer
from .decisions import to_internal, to_public, local_images
# List transitive files explicitly for Transformers' local dynamic loader.
# 本地动态加载需要显式列出间接依赖，否则搬移后可能缺少这些模块。
from .batching import collate_branches
from .schema import candidates
from .data_types import QuestionSpec


class ValenQwenForDecisionMaking(Qwen3_5PreTrainedModel):
    config_class = ValenQwenConfig
    base_model_prefix = "backbone"
    model_name = "Valen"
    architecture = "qwen"
    # The trained Mixer stays FP32 with a BF16 backbone. / BF16 基座、FP32 决策头。
    _keep_in_fp32_modules_strict = ["head"]

    def __init__(self, config, backbone=None, head=None):
        super().__init__(config)
        if (backbone is None) != (head is None):
            raise ValueError("Supply both a merged backbone and a trained head")
        self.backbone = backbone if backbone is not None else Qwen3_5Model(config)
        self.head = head if head is not None else build_head(config.text_config.hidden_size, config.valen_head)
        self.head.float()
        # Preserve full-SFT FP32 weights with BF16 compute. / 保留全参 SFT 的存储与计算精度。
        self.backbone_autocast_dtype = (
            torch.bfloat16 if getattr(config, "valen_backbone_autocast_dtype", None) == "bfloat16" else None
        )
        self._processor = None
        # Supplied weights are already trained: do not reinitialize them.
        # 传入的参数已训练完成，不能再次调用初始化覆盖它们。
        if backbone is not None:
            for module in self.modules():
                module._is_hf_initialized = True
        self.post_init()

    def get_input_embeddings(self):
        return self.backbone.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.backbone.set_input_embeddings(value)

    def forward(self, question=None, **inputs):
        if question is None:
            return self.backbone(**inputs)
        if hasattr(question, "questions"):
            return self.forward_state(question)
        return ValenQwen.forward(self, question)

    # Use the exact trained readouts and batching implementation.
    # 保留训练时的特征读取和 batch 逻辑。
    extract_features = ValenQwen.extract_features
    extract_state_features = ValenQwen.extract_state_features
    _state_features = ValenQwen._state_features
    forward_state = ValenQwen.forward_state
    forward_state_batch = ValenQwen.forward_state_batch
    forward_batch = ValenQwen.forward_batch
    score_features = ValenQwen.score_features
    # Earlier LoRA runtime snapshots do not use this helper. / 兼容旧 LoRA 运行时代码。
    if hasattr(ValenQwen, "_hidden"):
        _hidden = ValenQwen._hidden

    def make_compiler(self, media_root=".", execution=None, processor=None):
        if processor is None:
            if self._processor is None:
                self._processor = AutoProcessor.from_pretrained(self.config._name_or_path, trust_remote_code=True)
            processor = self._processor
        return Compiler(processor, media_root, self.config.valen_max_length, self.config.valen_media_kwargs,
                        execution=execution or self.config.qwen_execution)

    @torch.inference_mode()
    def predict(self, request, *, media_root=".", execution=None, temperature=1.0, processor=None):
        self.eval()
        # Only request content reaches the compiler. / 只编译输入，不读取标签。
        request = request["request"] if "request" in request else request
        temperature = request.get("temperature", temperature)
        public_request = request if "input" in request else None
        if public_request is not None:
            request = to_internal(public_request, allow_video=True)
        compiler = self.make_compiler(media_root, execution, processor)
        with local_images(request) as prepared:
            state = compiler.compile({"request": prepared})
            logits = self.forward_state(state) if state.inputs is not None else [self(q) for q in state.questions]
        answers = {q.qid: answer(q, z, temperature[q.kind] if isinstance(temperature, dict) else temperature)
                   for q, z in zip(state.questions, logits)}
        response = {"model": self.model_name, "answers": answers,
                "usage": {"input_tokens": state.logical_tokens, "output_tokens": 0},
                "internal_usage": {"compute_tokens": state.compute_tokens}}
        return to_public(public_request, response) if public_request is not None else response
