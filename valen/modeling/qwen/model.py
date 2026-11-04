from contextlib import nullcontext
import torch
from torch import nn
from valen import MODEL_NAME
from .heads import DecisionHead, build_head
from .batching import branch_batches, collate_branches
from .block_attention import enable as enable_block_attention


class ValenQwen(nn.Module):
    """Qwen3.5 backbone with a shared candidate decision head."""

    model_name = MODEL_NAME
    architecture = "qwen"

    def __init__(self, backbone, projection_dim=256, head_config=None, head=None):
        super().__init__()
        self.backbone = backbone
        self.head = head if head is not None else build_head(backbone.config.text_config.hidden_size,
                               dict({"projection_dim": projection_dim}, **(head_config or {})))
        self.backbone_autocast_dtype = None
        self.full_attention_mask = "causal"

    def _hidden(self, inputs, branches=None):
        # 保留 FP32 参数更新，前向使用 BF16。 / FP32 updates with BF16 backbone compute.
        device = next(self.backbone.parameters()).device
        context = (torch.autocast(device.type, dtype=self.backbone_autocast_dtype)
                   if self.backbone_autocast_dtype is not None else nullcontext())
        kwargs = {}
        if getattr(self, "full_attention_mask", "causal") == "block_bidirectional":
            # 前缀为 state，后缀为题目区；每行一个边界。 / One state/question boundary per row.
            enable_block_attention(self.backbone)
            kwargs["valen_prefix_lengths"] = torch.tensor([b.context_span[1] for b in branches], device=device)
        with context:
            return self.backbone(**inputs, use_cache=False, return_dict=True, **kwargs).last_hidden_state

    def forward(self, question):
        device = next(self.backbone.parameters()).device
        outputs = []
        for branch in question.branches:
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in branch.inputs.items()}
            hidden = self._hidden(inputs, [branch])
            outputs.append(self.head.score_features(self.head.extract_features(hidden, branch)))
        return torch.cat(outputs)

    def extract_features(self, question):
        """缓存投影前的特征。 / Cache raw readouts before trainable head layers."""
        device = next(self.backbone.parameters()).device
        features = []
        for branch in question.branches:
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in branch.inputs.items()}
            hidden = self._hidden(inputs, [branch])
            features.append(self.head.extract_features(hidden, branch))
        return features

    def extract_state_features(self, state):
        """一次编码 state，读取各题特征。 / One backbone graph for every question."""
        if not state.questions:
            return []
        if state.inputs is None:
            raise ValueError("Shared-state features require compiled shared inputs")
        device = next(self.backbone.parameters()).device
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in state.inputs.items()}
        hidden = self._hidden(inputs, [state.questions[0].branches[0]])
        return self._state_features(hidden, state)

    def _state_features(self, hidden, state):
        return [[self.head.extract_features(hidden, branch) for branch in q.branches]
                for q in state.questions]

    def forward_state(self, state):
        return [self.score_features(features) for features in self.extract_state_features(state)]

    def forward_state_batch(self, states, max_tokens=32768):
        """不同 state 组成 batch，同一 state 的题目共用一行。 / One row per shared state."""
        device = next(self.backbone.parameters()).device
        pad_token_id = getattr(self.backbone.config.text_config, "pad_token_id", None) or 0
        outputs = [[] for _ in states]
        representatives = []
        indices = []
        for index, state in enumerate(states):
            if not state.questions:
                continue
            if state.inputs is None or any(branch.inputs is not state.inputs
                    for q in state.questions for branch in q.branches):
                raise ValueError("Shared-state batch requires one shared input per state")
            representatives.append(state.questions[0])
            indices.append(index)
        for batch in branch_batches(representatives, max_tokens):
            inputs = collate_branches([branch for _, branch in batch], pad_token_id)
            inputs = {key: value.to(device) for key, value in inputs.items()}
            hidden = self._hidden(inputs, [branch for _, branch in batch])
            for row, (index, _) in enumerate(batch):
                original = indices[index]
                features = self._state_features(hidden[row:row + 1], states[original])
                outputs[original] = [self.score_features(f) for f in features]
        return outputs

    def forward_batch(self, questions, max_tokens=32768):
        """Forward several QAs together; each keeps its own decision head inputs.

        多条 QA 并行；每条保留独立的候选、角色区间和 Score 分支。
        """
        device = next(self.backbone.parameters()).device
        pad_token_id = getattr(self.backbone.config.text_config, "pad_token_id", None) or 0
        outputs = [[] for _ in questions]
        for batch in branch_batches(questions, max_tokens):
            inputs = collate_branches([branch for _, branch in batch], pad_token_id)
            inputs = {key: value.to(device) for key, value in inputs.items()}
            hidden = self._hidden(inputs, [branch for _, branch in batch])
            for row, (question_index, branch) in enumerate(batch):
                features = self.head.extract_features(hidden[row:row + 1], branch)
                outputs[question_index].append(self.head.score_features(features))
        return [torch.cat(branches) for branches in outputs]

    def score_features(self, features, head=None):
        head = self.head if head is None else head
        return torch.cat([head.score_features(hidden) for hidden in features])
