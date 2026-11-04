"""Grounding head for the ``bbox`` question type. / ``bbox`` 题的定位头。

Localization is a decision over image tokens: the question's decision readout scores
every visual token of the referenced image (softmax over tokens), and every token
regresses the distances from its cell centre to the four box edges (FCOS-style).
The predicted box is the regression of the most probable token; the confidence is
the probability mass of tokens whose centres fall inside that box.

定位即在图像 token 上做决策：读出位置为每个视觉 token 打分（token 上的 softmax），
每个 token 回归其格点中心到框四边的距离。预测框取概率最大的 token 的回归结果，
置信度为中心落在预测框内的 token 概率质量。坐标均按原图宽高归一化到 [0, 1]。
"""
from dataclasses import dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F


def grounding_signature(config):
    """None when disabled. / 未启用时为 None。"""
    settings = config.get("grounding_head")
    if not settings:
        return None
    settings = {} if settings is True else dict(settings)
    values = {"width": settings.get("width", 512), "hidden": settings.get("hidden", 1024)}
    if any(type(v) is not int or v <= 0 for v in values.values()):
        raise ValueError("grounding_head dimensions must be positive integers")
    return values


@dataclass
class BoxFeatures:
    decision: torch.Tensor  # [H]
    tokens: torch.Tensor    # [N, H]
    grid: tuple             # (rows, cols)


@dataclass
class BoxOutput:
    logits: torch.Tensor    # [N]
    boxes: torch.Tensor     # [N, 4] normalized x1, y1, x2, y2
    grid: tuple

    def detach(self):
        return BoxOutput(self.logits.detach(), self.boxes.detach(), self.grid)

    def decode(self):
        return decode(self)


def cell_centres(grid, device):
    rows, cols = grid
    y = (torch.arange(rows, device=device, dtype=torch.float32) + 0.5) / rows
    x = (torch.arange(cols, device=device, dtype=torch.float32) + 0.5) / cols
    cy, cx = torch.meshgrid(y, x, indexing="ij")
    return cx.reshape(-1), cy.reshape(-1)


class GroundingHead(nn.Module):
    def __init__(self, hidden_size, width=512, hidden=1024):
        super().__init__()
        self.decision_norm = nn.LayerNorm(hidden_size)
        self.token_norm = nn.LayerNorm(hidden_size)
        self.query = nn.Sequential(nn.Linear(hidden_size, hidden), nn.GELU(), nn.Linear(hidden, width))
        self.key = nn.Linear(hidden_size, width)
        self.gate = nn.Linear(hidden_size, width)
        self.regress = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, hidden), nn.GELU(), nn.Linear(hidden, 4))
        self.scale = math.sqrt(width)

    def extract_features(self, hidden, branch):
        if branch.image_positions is None or branch.image_grid is None:
            raise ValueError("bbox questions require compiled image token positions")
        return BoxFeatures(hidden[0, branch.decision_position], hidden[0, branch.image_positions.to(hidden.device)],
                           tuple(branch.image_grid))

    def forward(self, features):
        rows, cols = features.grid
        if features.tokens.shape[0] != rows * cols:
            raise ValueError("Image tokens do not match their grid")
        d = self.decision_norm(features.decision.float())
        t = self.token_norm(features.tokens.float())
        q = self.query(d)
        k = self.key(t)
        logits = (k @ q) / self.scale
        # Query-conditioned regression of edge distances in cell units. / 以格点为单位回归边距。
        distances = F.softplus(self.regress(k * torch.sigmoid(self.gate(d)))) + 0.5
        cx, cy = cell_centres(features.grid, logits.device)
        cell = torch.tensor([1 / cols, 1 / rows, 1 / cols, 1 / rows], device=logits.device)
        d = distances * cell
        boxes = torch.stack((cx - d[:, 0], cy - d[:, 1], cx + d[:, 2], cy + d[:, 3]), -1).clamp(0, 1)
        return BoxOutput(logits, boxes, features.grid)


def generalized_iou(a, b):
    """Elementwise GIoU of [N, 4] boxes. / 逐元素 GIoU。"""
    lt, rb = torch.maximum(a[:, :2], b[:, :2]), torch.minimum(a[:, 2:], b[:, 2:])
    inter = (rb - lt).clamp(min=0).prod(-1)
    area_a = (a[:, 2:] - a[:, :2]).clamp(min=0).prod(-1)
    area_b = (b[:, 2:] - b[:, :2]).clamp(min=0).prod(-1)
    union = area_a + area_b - inter
    iou = inter / union.clamp(min=1e-9)
    hull = (torch.maximum(a[:, 2:], b[:, 2:]) - torch.minimum(a[:, :2], b[:, :2])).clamp(min=0).prod(-1)
    return iou - (hull - union) / hull.clamp(min=1e-9), iou


def token_targets(target, grid, device):
    """Soft distribution over tokens for a box or point. / 框或点在 token 上的软目标。"""
    rows, cols = grid
    cx, cy = cell_centres(grid, device)
    if "bbox" in target:
        x1, y1, x2, y2 = target["bbox"]
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        sx, sy = max((x2 - x1) / 4, 0.5 / cols), max((y2 - y1) / 4, 0.5 / rows)
        inside = (cx >= x1) & (cx <= x2) & (cy >= y1) & (cy <= y2)
    else:
        mx, my = target["point"]
        sx, sy = 1 / cols, 1 / rows
        inside = torch.ones_like(cx, dtype=torch.bool)
    weights = torch.exp(-0.5 * (((cx - mx) / sx) ** 2 + ((cy - my) / sy) ** 2)) * inside
    if weights.sum() <= 0:
        # Tiny boxes between cell centres: the nearest token. / 极小框取最近 token。
        weights = torch.zeros_like(cx)
        weights[((cx - mx) ** 2 + (cy - my) ** 2).argmin()] = 1
    return weights / weights.sum()


def box_loss(output, target, l1_weight=5.0, giou_weight=2.0):
    weights = token_targets(target, output.grid, output.logits.device)
    ce = -(weights * output.logits.float().log_softmax(-1)).sum()
    terms = {"box_ce": ce.detach()}
    loss = ce
    if "bbox" in target:
        gold = torch.tensor(target["bbox"], device=output.boxes.device, dtype=torch.float32)
        active = weights > 0
        predicted = output.boxes[active].float()
        w = weights[active]
        l1 = (w * (predicted - gold).abs().sum(-1)).sum()
        giou, _ = generalized_iou(predicted, gold.expand_as(predicted))
        g = (w * (1 - giou)).sum()
        loss = loss + l1_weight * l1 + giou_weight * g
        terms.update(box_l1=l1.detach(), box_giou=g.detach())
    return loss, terms


@torch.no_grad()
def decode(output):
    """Box, point and confidence. / 预测框、点和置信度。"""
    probabilities = output.logits.float().softmax(-1)
    index = int(probabilities.argmax())
    box = output.boxes[index].float()
    cx, cy = cell_centres(output.grid, box.device)
    inside = (cx >= box[0]) & (cx <= box[2]) & (cy >= box[1]) & (cy <= box[3])
    confidence = float(probabilities[inside].sum()) if bool(inside.any()) else float(probabilities[index])
    x1, y1, x2, y2 = (round(float(v), 4) for v in box)
    return {"bbox": [x1, y1, x2, y2], "point": [round((x1 + x2) / 2, 4), round((y1 + y2) / 2, 4)],
            "confidence": round(min(max(confidence, 0.0), 1.0), 4)}
