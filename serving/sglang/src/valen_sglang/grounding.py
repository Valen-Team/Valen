"""Valen-Decision-1.5 grounding head for ``bbox`` questions.

The question's decision readout scores every visual token of the referenced image,
and every token regresses the distances from its cell centre to the four box edges.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn


def image_grids(processor, sizes):
    """Merged (rows, cols) token grid of each (height, width) image."""
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

    factor = processor.patch_size * processor.merge_size
    grids = []
    for height, width in sizes:
        height, width = smart_resize(
            height,
            width,
            factor,
            min_pixels=processor.size["shortest_edge"],
            max_pixels=processor.size["longest_edge"],
        )
        grids.append((height // factor, width // factor))
    return grids


def cell_centres(grid, device):
    rows, cols = grid
    y = (torch.arange(rows, device=device, dtype=torch.float32) + 0.5) / rows
    x = (torch.arange(cols, device=device, dtype=torch.float32) + 0.5) / cols
    cy, cx = torch.meshgrid(y, x, indexing="ij")
    return cx.reshape(-1), cy.reshape(-1)


class GroundingHead(nn.Module):
    def __init__(self, hidden_size, width=512, hidden=1024):
        super().__init__()
        fp32 = {"dtype": torch.float32}
        self.decision_norm = nn.LayerNorm(hidden_size, **fp32)
        self.token_norm = nn.LayerNorm(hidden_size, **fp32)
        self.query = nn.Sequential(
            nn.Linear(hidden_size, hidden, **fp32),
            nn.GELU(),
            nn.Linear(hidden, width, **fp32),
        )
        self.key = nn.Linear(hidden_size, width, **fp32)
        self.gate = nn.Linear(hidden_size, width, **fp32)
        self.regress = nn.Sequential(
            nn.LayerNorm(width, **fp32),
            nn.Linear(width, hidden, **fp32),
            nn.GELU(),
            nn.Linear(hidden, 4, **fp32),
        )
        self.scale = math.sqrt(width)

    def forward(self, decision, tokens, grid):
        """Box x1, y1, x2, y2 and confidence, normalized to the image."""
        rows, cols = grid
        d = self.decision_norm(decision.float())
        k = self.key(self.token_norm(tokens.float()))
        probabilities = ((k @ self.query(d)) / self.scale).softmax(-1)
        # Query-conditioned regression of edge distances in cell units.
        distances = F.softplus(self.regress(k * torch.sigmoid(self.gate(d)))) + 0.5
        cx, cy = cell_centres(grid, k.device)
        index = probabilities.argmax()
        left, top, right, bottom = distances[index] * k.new_tensor(
            [1 / cols, 1 / rows, 1 / cols, 1 / rows]
        )
        box = torch.stack(
            (cx[index] - left, cy[index] - top, cx[index] + right, cy[index] + bottom)
        ).clamp(0, 1)
        # Confidence is the probability mass of tokens centred inside the box.
        inside = (cx >= box[0]) & (cx <= box[2]) & (cy >= box[1]) & (cy <= box[3])
        confidence = probabilities[inside].sum() if inside.any() else probabilities[index]
        return torch.cat((box, confidence.reshape(1)))


def box_answer(values):
    """The native bbox answer from the head's five values."""
    x1, y1, x2, y2 = (round(v, 4) for v in values[:4])
    return {
        "type": "bbox",
        "bbox": [x1, y1, x2, y2],
        "point": [round((x1 + x2) / 2, 4), round((y1 + y2) / 2, 4)],
        "confidence": round(min(max(values[4], 0.0), 1.0), 4),
    }
