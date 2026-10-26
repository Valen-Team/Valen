"""Language-conditioned spatial queries, coordinate logits and mask decoding."""
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from .model import ValenQwen


def spatial_signature(config):
    result = {key: config.get(key, default) for key, default in (
        ("task", "det"), ("spatial_width", 256), ("spatial_queries", 100),
        ("spatial_layers", 2), ("spatial_heads", 8), ("coordinate_bins", 512))}
    if result["task"] not in {"det", "seg", "gui"}:
        raise ValueError("Spatial task must be det, seg or gui")
    for key, value in result.items():
        if key != "task" and (type(value) is not int or value <= 0):
            raise ValueError(f"{key} must be a positive integer")
    if result["spatial_width"] % result["spatial_heads"] or result["coordinate_bins"] < 2:
        raise ValueError("Invalid spatial attention width or coordinate bins")
    return result


@dataclass
class SpatialOutput:
    class_logits: torch.Tensor
    coordinate_logits: torch.Tensor | None = None
    boxes: torch.Tensor | None = None
    mask_logits: torch.Tensor | None = None


class SpatialHead(nn.Module):
    def __init__(self, hidden_size, config):
        super().__init__()
        self.settings = spatial_signature(config)
        width = self.settings["spatial_width"]
        count = self.settings["spatial_queries"] if self.settings["task"] == "det" else 1
        self.input_projection = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, width))
        self.position = nn.Sequential(nn.Linear(2, width), nn.GELU(), nn.Linear(width, width))
        self.queries = nn.Parameter(torch.randn(count, width) / math.sqrt(width))
        layer = nn.TransformerDecoderLayer(width, self.settings["spatial_heads"], 4 * width,
                                          dropout=0.0, activation="gelu", batch_first=True,
                                          norm_first=True)
        self.decoder = nn.TransformerDecoder(layer, self.settings["spatial_layers"], nn.LayerNorm(width))
        self.category_query = nn.Linear(width, width, bias=False)
        self.category_key = nn.Linear(width, width, bias=False)
        self.background = nn.Linear(width, 1)
        if self.settings["task"] == "seg":
            self.mask_features = nn.Sequential(
                nn.ConvTranspose2d(width, width // 2, 2, stride=2), nn.GELU(),
                nn.ConvTranspose2d(width // 2, width // 4, 2, stride=2), nn.GELU())
            self.mask_query = nn.Linear(width, width // 4)
        else:
            self.coordinate = nn.Linear(width, 4 * self.settings["coordinate_bins"])

    def forward(self, image_features, decision, category_features, grid):
        h, w = grid
        if len(image_features) != h * w:
            raise ValueError("Image tokens do not match their spatial grid")
        image = self.input_projection(image_features.float())
        context = self.input_projection(decision.float())
        categories = self.input_projection(category_features.float())
        y, x = torch.meshgrid(torch.arange(h, device=image.device),
                              torch.arange(w, device=image.device), indexing="ij")
        position = torch.stack(((x + .5) / w, (y + .5) / h), -1).reshape(-1, 2)
        memory = image + self.position(position)
        queries = self.decoder((self.queries + context).unsqueeze(0), memory.unsqueeze(0))[0]
        scores = self.category_query(queries) @ self.category_key(categories).T / math.sqrt(image.shape[-1])
        classes = torch.cat((scores, self.background(queries)), -1)
        if self.settings["task"] == "seg":
            features = self.mask_features(image.T.reshape(1, -1, h, w))[0]
            masks = torch.einsum("qc,chw->qhw", self.mask_query(queries), features)
            return SpatialOutput(classes, mask_logits=masks)
        bins = self.settings["coordinate_bins"]
        logits = self.coordinate(queries).view(-1, 4, bins)
        support = torch.linspace(0, 1, bins, device=logits.device)
        cx, cy, bw, bh = (logits.softmax(-1) * support).sum(-1).unbind(-1)
        boxes = torch.stack((cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2), -1).clamp(0, 1)
        return SpatialOutput(classes, logits, boxes)


class ValenSpatial(ValenQwen):
    architecture = "qwen_spatial"

    def __init__(self, backbone, config):
        nn.Module.__init__(self)
        self.backbone = backbone
        self.head = SpatialHead(backbone.config.text_config.hidden_size, config)
        self.backbone_autocast_dtype = None
        self.spatial_config = spatial_signature(config)
        self.model_name = "Valen-Decision-" + {"det": "Det", "seg": "Seg", "gui": "GUI"}[config["task"]]

    def forward(self, question):
        device = next(self.backbone.parameters()).device
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in question.inputs.items()}
        hidden = self._hidden(inputs)[0]
        image = hidden[inputs["input_ids"][0] == self.backbone.config.image_token_id]
        categories = torch.stack([hidden[start:end].float().mean(0) for start, end in question.category_spans])
        return self.head(image, hidden[question.decision_position], categories, question.grid)

    def forward_state(self, state):
        return [self(question) for question in state.questions]


def decode_spatial(question, output, score_threshold=0.05):
    scores, labels = output.class_logits.float().softmax(-1)[:, :-1].max(-1)
    result = {"type": question.kind, "image_size": list(question.original_size)}
    if output.mask_logits is not None:
        logits = F.interpolate(output.mask_logits[:1, None], size=question.original_size,
                               mode="bilinear", align_corners=False)[0, 0]
        result.update(mask=logits > 0, confidence=float(scores[0]))
        return result
    order = scores.argsort(descending=True)
    if question.metadata.get("allow_empty_target", False):
        order = order[output.class_logits.argmax(-1)[order] != len(question.categories)]
    if question.kind != "det" or question.metadata.get("referring", False):
        order = order[:1]
    else:
        order = order[scores[order] >= score_threshold]
    boxes = []
    for index in order.tolist():
        box = output.boxes[index].detach().float().cpu().tolist()
        if box[0] < box[2] and box[1] < box[3]:
            boxes.append({"bbox": box, "label": int(labels[index]),
                          "category": question.categories[int(labels[index])], "score": float(scores[index])})
    result["boxes"] = boxes
    if question.kind == "gui" and boxes:
        x1, y1, x2, y2 = boxes[0]["bbox"]
        result["point"] = [(x1 + x2) / 2, (y1 + y2) / 2]
    return result


def spatial_response(question, output, score_threshold=0.05):
    """Convert spatial predictions to a JSON-serializable response."""
    result = decode_spatial(question, output, score_threshold)
    if "mask" in result:
        import numpy as np
        from pycocotools import mask as mask_api
        mask = result["mask"].detach().cpu().numpy().astype("uint8")
        rle = mask_api.encode(np.asfortranarray(mask))
        result["mask"] = {"size": rle["size"], "counts": rle["counts"].decode("ascii")}
    return result
