"""Set matching for coordinate distributions and binary referring-mask losses."""
import torch
from torch.nn import functional as F

from valen.data.spatial import decode_segmentation_target


def box_iou(boxes, targets, generalized=False):
    intersection = (torch.minimum(boxes[:, None, 2:], targets[None, :, 2:])
                    - torch.maximum(boxes[:, None, :2], targets[None, :, :2])).clamp(min=0).prod(-1)
    area = (boxes[:, 2:] - boxes[:, :2]).clamp(min=0).prod(-1)
    target_area = (targets[:, 2:] - targets[:, :2]).clamp(min=0).prod(-1)
    union = area[:, None] + target_area[None] - intersection
    iou = intersection / union.clamp(min=1e-7)
    if not generalized:
        return iou
    cover = (torch.maximum(boxes[:, None, 2:], targets[None, :, 2:])
             - torch.minimum(boxes[:, None, :2], targets[None, :, :2])).clamp(min=0).prod(-1)
    return iou - (cover - union) / cover.clamp(min=1e-7)


class SpatialObjective:
    def __init__(self, config):
        self.config = config

    def __call__(self, output, question):
        target = question.target
        if target is None:
            raise ValueError("Spatial SFT requires labels")
        device = output.class_logits.device
        if output.mask_logits is not None:
            truth, valid = decode_segmentation_target(target, question.media_root)
            if tuple(truth.shape) != question.original_size:
                raise ValueError("Segmentation mask is not aligned to the original image")
            present = int(bool(truth[valid].any()))
            truth = F.interpolate(truth[None, None], size=output.mask_logits.shape[-2:],
                                  mode="nearest").to(device)[0]
            probability = output.mask_logits.sigmoid()
            if target.get("ignore_mask") is None:
                bce = F.binary_cross_entropy_with_logits(output.mask_logits, truth)
                dice = 1 - (2 * (probability * truth).sum() + 1) / (probability.sum() + truth.sum() + 1)
            else:
                valid = F.interpolate(valid.float()[None, None], size=output.mask_logits.shape[-2:],
                                      mode="nearest").to(device)[0]
                bce = (F.binary_cross_entropy_with_logits(output.mask_logits, truth, reduction="none")
                       * valid).sum() / valid.sum().clamp(min=1)
                dice = 1 - (2 * (probability * truth * valid).sum() + 1) / (
                    (probability * valid).sum() + (truth * valid).sum() + 1)
            label = torch.tensor([0 if present else len(question.categories)], device=device)
            classification = (F.cross_entropy(output.class_logits, label) if valid.any()
                              else output.class_logits.sum() * 0)
            loss = bce + self.config.get("dice_weight", 1.0) * dice + .1 * classification
            return loss, {"bce": float(bce.detach()), "dice": float(dice.detach()),
                          "classification": float(classification.detach())}
        from scipy.optimize import linear_sum_assignment
        boxes = torch.tensor(target.get("boxes", []), device=device, dtype=torch.float32).reshape(-1, 4)
        labels = torch.tensor(target.get("labels", []), device=device, dtype=torch.long)
        predictions = output.boxes
        if len(boxes) > len(predictions):
            raise ValueError("More ground-truth boxes than spatial queries; increase spatial_queries")
        background = len(question.categories)
        assigned = torch.full((len(predictions),), background, device=device, dtype=torch.long)
        zero = output.coordinate_logits.sum() * 0
        coordinate_loss = l1 = giou = zero
        if len(boxes):
            with torch.no_grad():
                cost = -output.class_logits.softmax(-1)[:, labels]
                cost = cost + 5 * torch.cdist(predictions, boxes, p=1)
                cost = cost - 2 * box_iou(predictions, boxes, generalized=True)
                rows, columns = linear_sum_assignment(cost.detach().float().cpu().numpy())
                rows = torch.as_tensor(rows, device=device)
                columns = torch.as_tensor(columns, device=device)
            matched = boxes[columns]
            assigned[rows] = labels[columns]
            centers = (matched[:, :2] + matched[:, 2:]) / 2
            sizes = matched[:, 2:] - matched[:, :2]
            bins = output.coordinate_logits.shape[-1]
            coordinates = (torch.cat((centers, sizes), -1) * (bins - 1)).clamp(0, bins - 1)
            lower = coordinates.floor().long()
            upper = (lower + 1).clamp(max=bins - 1)
            fraction = coordinates - lower
            log_probabilities = output.coordinate_logits[rows].log_softmax(-1)
            coordinate_loss = -((1 - fraction) * log_probabilities.gather(-1, lower[..., None]).squeeze(-1)
                                + fraction * log_probabilities.gather(-1, upper[..., None]).squeeze(-1)).mean()
            l1 = F.l1_loss(predictions[rows], matched)
            giou = (1 - box_iou(predictions[rows], matched, generalized=True).diag()).mean()
        weights = torch.ones(background + 1, device=device)
        weights[-1] = self.config.get("no_object_weight", .1)
        classification = F.cross_entropy(output.class_logits, assigned, weight=weights)
        loss = classification + coordinate_loss + self.config.get("box_l1_weight", 5.0) * l1
        loss = loss + self.config.get("box_giou_weight", 2.0) * giou
        return loss, {"classification": float(classification.detach()),
                      "coordinate": float(coordinate_loss.detach()), "l1": float(l1.detach()),
                      "giou": float(giou.detach())}
