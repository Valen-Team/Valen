from copy import deepcopy

from ..interfaces import ArchitectureBackend, Capabilities, Decision, PolicyInput


class QwenBackend(ArchitectureBackend):
    name = "qwen"
    unit_scope = "question"
    capabilities = Capabilities(video=True, frozen_feature_cache=True)

    def build_model(self, config):
        from .builder import build_qwen
        return build_qwen(config)

    def build_compiler(self, config, media_root):
        from transformers import AutoProcessor
        from valen.data.compilers.qwen import Compiler
        from .pretrained import published_config
        return Compiler(AutoProcessor.from_pretrained(config["model_path"], local_files_only=True,
                        trust_remote_code=published_config(config["model_path"]) is not None),
                        media_root, config.get("max_length", 8192), config.get("media_kwargs"),
                        execution=config.get("qwen_execution", "question"))

    def candidates(self, question):
        from valen.data.schema import candidates
        return candidates(question)

    def optimizer_groups(self, model, config):
        from .builder import qwen_optimizer_groups
        return qwen_optimizer_groups(model, config)

    def inference_units(self, model, state):
        if getattr(state, "inputs", None) is not None:
            if state.questions:
                yield [Decision(q, z) for q, z in zip(state.questions, model.forward_state(state))]
            return
        for question in state.questions:
            yield [Decision(question, model(question))]

    def question_logits(self, model, question, rollout=None):
        features = getattr(rollout, "features", None)
        return model.score_features(features) if features is not None else model(question)

    def training_units(self, model, state, rollouts):
        if len(rollouts) != len(state.questions):
            raise ValueError("Rollout/question count mismatch")
        if getattr(state, "inputs", None) is not None:
            cached = [getattr(rollout, "features", None) for rollout in rollouts]
            if any(features is not None for features in cached):
                if not all(features is not None for features in cached):
                    raise ValueError("Shared state requires all or none of its features cached")
                logits = [model.score_features(features) for features in cached]
            else:
                logits = model.forward_state(state)
            if state.questions:
                yield [Decision(q, z, r) for q, z, r in zip(state.questions, logits, rollouts)]
            return
        for question, rollout in zip(state.questions, rollouts):
            # Yield before the next forward, allowing immediate backward and release.
            yield [Decision(question, self.question_logits(model, question, rollout), rollout)]

    def policy_inputs(self, model, state, reference, cache_features=False):
        if getattr(state, "inputs", None) is not None:
            features = model.extract_state_features(state) if cache_features else [None] * len(state.questions)
            logits = [model.score_features(f) for f in features] if cache_features else model.forward_state(state)
            refs = [None] * len(state.questions)
            if reference is not None:
                refs = ([model.score_features(f, reference) for f in features] if cache_features
                        else reference.forward_state(state))
            for q, z, ref, f in zip(state.questions, logits, refs, features):
                yield PolicyInput(q, z, ref, f)
            return
        for question in state.questions:
            features = model.extract_features(question) if cache_features else None
            logits = model.score_features(features) if cache_features else model(question)
            ref = None
            if reference is not None:
                ref = model.score_features(features, reference) if cache_features else reference(question)
            yield PolicyInput(question, logits, ref, features)

    def validate_state(self, config, state):
        if config.get("stage") == "text" and any(not q.is_text for q in state.questions):
            raise ValueError("Text stage requires a text-only dataset")

    def validate_feature_cache(self, model):
        if not hasattr(model, "extract_features") or any(p.requires_grad for p in model.backbone.parameters()):
            raise ValueError("Feature reuse requires a fully frozen Valen backbone")
        if any(not n.startswith("head.") for n, p in model.named_parameters() if p.requires_grad):
            raise ValueError("Feature reuse requires only head parameters to be trainable")

    def make_reference(self, model, weights, cache_features=False):
        if not cache_features:
            return super().make_reference(model, weights)
        self.validate_feature_cache(model)
        reference = deepcopy(model.head)
        reference.load_state_dict({k.removeprefix("head."): v for k, v in weights.items()})
        return reference.requires_grad_(False).eval()

    def base_paths(self, config):
        return config["model_path"]

    def validate_initialization(self, previous, config):
        super().validate_initialization(previous, config)
        from .heads import head_signature
        if head_signature(previous) != head_signature(config):
            raise ValueError("Initialization backbone/head mismatch")
        if previous.get("full_attention_mask", "causal") != config.get("full_attention_mask", "causal"):
            raise ValueError("Initialization full_attention_mask mismatch")

    def adaptation(self, model, config):
        from .heads import head_signature
        stage = config.get("stage", "joint")
        full = config.get("finetuning_type", "lora") == "full" and stage == "joint"
        return {"stage": stage, "decision_head": head_signature(config),
                "finetuning_type": config.get("finetuning_type", "lora"),
                "parameter_dtype": str(next(model.parameters()).dtype),
                "backbone_autocast_dtype": str(getattr(model, "backbone_autocast_dtype", None)),
                "qwen_execution": config.get("qwen_execution", "question"),
                "full_attention_mask": config.get("full_attention_mask", "causal"),
                "text": "full" if full else "lora" if stage != "warmup" else "frozen",
                "vision_merger": stage in {"joint", "vision_top"},
                "vision_unfreeze_layers": len(model.backbone.visual.blocks) if full else 4 if stage == "vision_top" else 0,
                "vision": "full" if full else "partial" if stage == "vision_top" else "merger" if stage == "joint" else "frozen",
                "attention_backends": getattr(model, "attention_backends", None),
                "lora_targets": getattr(model, "lora_targets", [])}
