"""Native vLLM Qwen3.5 execution with Valen's FP32 Mixer readout."""

from torch import nn
from vllm.forward_context import get_forward_context
from vllm.model_executor.models.adapters import as_embedding_model
from vllm.model_executor.models.qwen3_5 import (
    Qwen3_5ForConditionalGeneration,
    Qwen3_5ProcessingInfo,
)
from vllm.model_executor.models.qwen3_vl import (
    Qwen3VLDummyInputsBuilder,
    Qwen3VLMultiModalProcessor,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.tokenizers.registry import cached_tokenizer_from_config

from valen_vllm.blocks import GLUE, block_attention, token_segments
from valen_vllm.head import ValenMixerHead
from valen_vllm.pooler import ValenPooler


class ValenProcessingInfo(Qwen3_5ProcessingInfo):
    def get_hf_processor(self, **kwargs):
        processor = super().get_hf_processor(**kwargs)
        image_kwargs = self.get_hf_config().valen_media_kwargs.get("images_kwargs", {})
        if "size" in image_kwargs:
            processor.image_processor.size = dict(image_kwargs["size"])
        return processor


class BlockAttention(nn.Module):
    """A full-attention layer's kernel under Valen-Decision-1.5 block masks.

    Without blocks the engine kernel runs unchanged. A block forward is one
    complete prefill, so it neither reads nor writes the KV cache.
    """

    def __init__(self, attn, scale):
        super().__init__()
        self.attn = attn
        self.scale = scale
        self.blocks = None

    def forward(self, query, key, value):
        if self.blocks is None:
            return self.attn(query, key, value)
        attn = self.attn
        q = query.view(-1, attn.num_heads, attn.head_size)
        k = key.view(-1, attn.num_kv_heads, attn.head_size)
        v = value.view(-1, attn.num_kv_heads, attn.head_size)
        output = q.new_zeros(q.shape)
        for start, end, segments in self.blocks:
            output[start:end] = block_attention(
                q[start:end], k[start:end], v[start:end], segments, self.scale
            )
        return output.view(-1, attn.num_heads * attn.head_size)


_Base = as_embedding_model(Qwen3_5ForConditionalGeneration)


@MULTIMODAL_REGISTRY.register_processor(
    Qwen3VLMultiModalProcessor,
    info=ValenProcessingInfo,
    dummy_inputs=Qwen3VLDummyInputsBuilder,
)
class ValenQwenForDecisionMaking(_Base):
    supports_multimodal_pruning = False
    # Block boundaries are read from the prompt's chat-turn tokens.
    requires_raw_input_tokens = True

    def __init__(self, *, vllm_config, prefix="", **kwargs):
        super().__init__(vllm_config=vllm_config, prefix=prefix, **kwargs)
        mask = vllm_config.model_config.hf_config.valen_full_attention_mask
        self.block_layers = []
        if mask == "causal":
            return
        if mask not in ("block_bidirectional", "bbox_blocks"):
            raise ValueError(f"valen_full_attention_mask={mask} is not served by vLLM")
        tokenizer = cached_tokenizer_from_config(vllm_config.model_config)
        self.glue = [tokenizer.encode(text, add_special_tokens=False) for text in GLUE]
        self.bidirectional = mask == "block_bidirectional"
        for layer in self.language_model.model.layers:
            if layer.layer_type == "full_attention":
                attention = layer.self_attn
                attention.attn = BlockAttention(attention.attn, attention.scaling)
                self.block_layers.append(attention.attn)

    def forward(
        self,
        input_ids,
        positions,
        intermediate_tensors=None,
        inputs_embeds=None,
        **kwargs,
    ):
        blocks = None
        metadata = self.block_layers and get_forward_context().attn_metadata
        if metadata:
            # One complete prompt per request: no prefix reuse or chunking.
            starts = metadata[self.block_layers[0].attn.layer_name].query_start_loc
            starts = starts.tolist()
            blocks = [
                (a, b, token_segments(input_ids[a:b], self.glue, self.bidirectional))
                for a, b in zip(starts, starts[1:])
                if b > a
            ]
            # Profiling prompts carry no chat turns and keep the engine kernel.
            if all(segments is None for _, _, segments in blocks):
                blocks = None
        for layer in self.block_layers:
            layer.blocks = blocks
        return super().forward(
            input_ids, positions, intermediate_tensors, inputs_embeds, **kwargs
        )

    def _init_pooler(self, vllm_config, prefix=""):
        config = vllm_config.model_config.hf_config
        return ValenPooler(
            ValenMixerHead(config.text_config.hidden_size, config.valen_head)
        )

    def load_weights(self, weights):
        from vllm.model_executor.model_loader.weight_utils import default_weight_loader
        from vllm.model_executor.models.qwen3_5 import Qwen3_5ForConditionalGeneration

        params = dict(self.pooler.head.named_parameters())
        seen = set()

        def backbone():
            for name, value in weights:
                if name.startswith("head."):
                    key = name.removeprefix("head.")
                    default_weight_loader(params[key], value)
                    seen.add(key)
                elif name.startswith("backbone."):
                    yield "model." + name.removeprefix("backbone."), value
                elif name.startswith("grounding."):
                    # The bbox head has no readout path in this plugin.
                    continue
                else:
                    yield name, value

        loaded = Qwen3_5ForConditionalGeneration.load_weights(self, backbone())
        return loaded | {"pooler.head." + key for key in seen}
