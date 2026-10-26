"""Spatial extension of Valen's grouped RLCD and confidence-calibration objective.

The latent action is NULL or (query, center-x bin, center-y bin, width bin,
height bin). Its exact joint probability and factorized reference KL include
the presence decision. Ground truth is used only after action sampling.
"""
from dataclasses import dataclass

import torch
from torch.nn import functional as F

from .rlcd import group_advantages, validate_options
from .spatial_loss import box_iou


@dataclass
class SpatialActions:
    present: torch.Tensor
    query: torch.Tensor
    coordinates: torch.Tensor


class SpatialPolicy:
    def __init__(self, output, allow_empty=False):
        if output.class_logits.ndim != 2 or output.class_logits.shape[-1] != 2:
            raise ValueError("Spatial RLCD requires one referring target category")
        if output.coordinate_logits is None or output.coordinate_logits.ndim != 3:
            raise ValueError("Spatial RLCD requires four categorical bbox coordinates")
        logits = output.class_logits.float()
        self.foreground = logits.softmax(-1)[:, 0]
        odds = logits[:, 0] - logits[:, 1]
        self.query = odds.log_softmax(-1)
        self.coordinates = output.coordinate_logits.float().log_softmax(-1)
        if self.coordinates.shape[:2] != (len(logits), 4):
            raise ValueError("Coordinate distributions must align with spatial queries")
        self.allow_empty = bool(allow_empty)
        if self.allow_empty:
            self.log_present = F.logsigmoid(odds.max())
            self.log_absent = F.logsigmoid(-odds.max())
        else:
            self.log_present = odds.sum()*0
            self.log_absent = odds.sum()*0

    def sample(self, count):
        query = torch.multinomial(self.query.exp(), count, replacement=True)
        probabilities = self.coordinates[query].exp()
        bins = torch.multinomial(probabilities.reshape(-1, probabilities.shape[-1]), 1).reshape(count, 4)
        present = (torch.rand(count, device=query.device) < self.log_present.exp()
                   if self.allow_empty else torch.ones(count, dtype=torch.bool, device=query.device))
        return SpatialActions(present, query, bins)

    def log_prob(self, actions):
        selected = self.coordinates[actions.query].gather(-1, actions.coordinates[..., None]).squeeze(-1).sum(-1)
        positive = self.log_present + self.query[actions.query] + selected
        if not self.allow_empty and not bool(actions.present.all()):
            raise ValueError("NULL actions are unavailable for this inference protocol")
        return torch.where(actions.present, positive, self.log_absent)

    def confidence(self, actions):
        return torch.where(actions.present, self.foreground[actions.query], -torch.expm1(self.log_present))

    def boxes(self, actions):
        support = actions.coordinates.float() / (self.coordinates.shape[-1] - 1)
        center, size = support[:, :2], support[:, 2:]
        return torch.cat((center-size/2, center+size/2), -1).clamp(0, 1)

    def kl(self, reference):
        if self.allow_empty != reference.allow_empty or self.coordinates.shape != reference.coordinates.shape:
            raise ValueError("Policy and fixed reference must use the same action support")
        query = self.query.exp()
        conditional = (query * (self.query-reference.query)).sum()
        conditional = conditional + (query[:, None, None] * self.coordinates.exp()
                                      * (self.coordinates-reference.coordinates)).sum()
        if not self.allow_empty:
            return conditional
        presence = self.log_present.exp()
        absent = self.log_absent.exp()
        bernoulli = presence*(self.log_present-reference.log_present) + absent*(self.log_absent-reference.log_absent)
        return bernoulli + presence*conditional

    def entropy(self):
        query = self.query.exp()
        conditional = -(query*self.query).sum() - (query[:, None, None]*self.coordinates.exp()*self.coordinates).sum()
        if not self.allow_empty:
            return conditional
        return (-self.log_present.exp()*self.log_present-self.log_absent.exp()*self.log_absent
                + self.log_present.exp()*conditional)


@dataclass
class SpatialRollout:
    actions: SpatialActions
    old_log_prob: torch.Tensor
    advantages: torch.Tensor
    rewards: torch.Tensor
    correctness: torch.Tensor
    iou: torch.Tensor
    reference: SpatialPolicy


class SpatialRLCDObjective:
    def __init__(self, options=None):
        self.options = validate_options(options or {})

    @staticmethod
    def policy(output, question):
        if question.kind != "det" or len(question.categories) != 1:
            raise ValueError("Spatial RLCD currently supports single-target referring Det only")
        return SpatialPolicy(output, question.metadata.get("allow_empty_target", False))

    @torch.no_grad()
    def rollout(self, output, reference_output, question):
        policy, reference = self.policy(output, question), self.policy(reference_output, question)
        actions = policy.sample(self.options["group_size"])
        target = question.target
        if target is None:
            raise ValueError("Spatial RLCD rewards require sealed training labels")
        boxes = torch.tensor(target.get("boxes", []), device=actions.query.device, dtype=torch.float32).reshape(-1, 4)
        if len(boxes):
            overlap = box_iou(policy.boxes(actions), boxes).max(-1).values
            overlap = torch.where(actions.present, overlap, torch.zeros_like(overlap))
            correct = actions.present & (overlap >= .5)
        else:
            correct = ~actions.present
            overlap = correct.float()
        correctness = correct.float()
        confidence = policy.confidence(actions)
        presence_correct = actions.present == bool(len(boxes))
        spatial_reward = torch.where(presence_correct, .5*overlap+.5*correctness, -torch.ones_like(overlap))
        reward = (self.options["correctness_weight"] * spatial_reward
                  - self.options["confidence_weight"] * (confidence-correctness).square())
        return SpatialRollout(actions, policy.log_prob(actions),
                              group_advantages(reward, self.options["advantage_epsilon"]),
                              reward, correctness, overlap, reference)

    def loss(self, output, question, rollout):
        policy = self.policy(output, question)
        ratio = (policy.log_prob(rollout.actions)-rollout.old_log_prob).exp()
        advantage = rollout.advantages
        direct = ratio*advantage
        clipped = ratio.clamp(1-self.options["clip_epsilon"], 1+self.options["clip_epsilon"])*advantage
        policy_loss = -torch.minimum(direct, clipped).mean()
        brier = (policy.confidence(rollout.actions)-rollout.correctness).square().mean()
        kl = policy.kl(rollout.reference)
        loss = policy_loss+self.options["brier_weight"]*brier+self.options["beta"]*kl
        metrics = dict(policy_loss=policy_loss, brier=brier, kl=kl, reward=rollout.rewards.mean(),
                       sample_correctness=rollout.correctness.mean(), sample_iou=rollout.iou.mean(),
                       entropy=policy.entropy(), reward_std=rollout.rewards.std(unbiased=False),
                       zero_advantage_group=(advantage==0).all().float(),
                       clip_fraction=(clipped<direct).float().mean())
        return loss, {key:float(value.detach()) for key,value in metrics.items()}
