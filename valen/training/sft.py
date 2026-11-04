import torch
from valen.modeling.factory import backend_for_model
from valen.modeling.qwen.grounding import box_loss


def question_loss(logits, target, kind, rps_weight=0.0, brier_weight=0.0):
    logits = logits.float()
    y = torch.as_tensor(target, dtype=torch.float32, device=logits.device)
    ce = -(y * logits.log_softmax(-1)).sum()
    rps = logits.new_zeros(())
    if kind == "score":
        rps = (logits.softmax(-1).cumsum(-1)[:-1] - y.cumsum(-1)[:-1]).square().mean()
    brier = (logits.softmax(-1) - y).square().sum()
    terms = {"ce": ce.detach(), "rps": rps.detach()}
    if brier_weight:
        terms["brier"] = brier.detach()
    return ce + rps_weight * rps + brier_weight * brier, terms


class SFTObjective:
    metric_names = ("ce", "rps", "box_ce", "box_l1", "box_giou")
    num_iterations = 1

    def __init__(self, config):
        self.num_iterations = config.get("sft_iterations", 1)
        if type(self.num_iterations) is not int or self.num_iterations < 1:
            raise ValueError("sft_iterations must be a positive integer")
        self.rps_weight = config.get("rps_weight", 0.0)
        self.brier_weight = config.get("brier_weight", 0.0)
        self.bbox_weight = config.get("bbox_loss_weight", 1.0)
        if self.brier_weight:
            self.metric_names = (*self.metric_names, "brier")

    def prepare(self, model, pack):
        return [[None for _ in state.questions] for state in pack]

    def loss_from_logits(self, logits, question, rollout=None):
        if question.kind == "bbox":
            loss, terms = box_loss(logits, question.target)
            return self.bbox_weight * loss, terms
        return question_loss(logits, question.target, question.kind, self.rps_weight, self.brier_weight)

    def loss(self, model, question, rollout):
        return self.loss_from_logits(backend_for_model(model).question_logits(model, question, rollout), question, rollout)

    def state_losses(self, model, state, rollouts):
        return [self.loss_from_logits(d.logits, d.question, d.rollout)
                for unit in backend_for_model(model).training_units(model, state, rollouts) for d in unit]

    def state_dict(self):
        return None
