"""Validated normalized-xyxy conversion and official COCO bbox scoring."""
import math


CONDITIONAL_BENCHMARK = "coco2017_conditional_val2017"
SINGLE_CATEGORY_PROTOCOL = "coco_single_category_v3"


def predictions_to_coco(question, prediction):
    height, width = question.original_size
    category_ids = question.metadata["category_ids"]
    if height <= 0 or width <= 0 or len(category_ids) != len(question.categories):
        raise ValueError("COCO dimensions/category mapping are inconsistent")
    result = []
    for item in prediction["boxes"]:
        x1, y1, x2, y2 = item["bbox"]
        score, label = item["score"], item["label"]
        if (not all(math.isfinite(v) for v in (x1, y1, x2, y2, score))
                or not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1)
                or not 0 <= score <= 1 or type(label) is not int or not 0 <= label < len(category_ids)):
            raise ValueError("Invalid normalized xyxy prediction, score or category index")
        result.append({"image_id": question.metadata["image_id"], "category_id": category_ids[label],
                       "score": score, "bbox": [x1 * width, y1 * height,
                                                 (x2 - x1) * width, (y2 - y1) * height]})
    return result


def score_coco(annotations, predictions, image_ids=None, use_categories=True):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    names = ("AP", "AP50", "AP75", "AP_S", "AP_M", "AP_L", "AR1", "AR10", "AR100")
    if not predictions:
        return dict.fromkeys(names, 0.0)
    truth = COCO(str(annotations))
    scorer = COCOeval(truth, truth.loadRes(predictions), "bbox")
    if image_ids is not None:
        scorer.params.imgIds = list(image_ids)
    scorer.params.useCats = int(use_categories)
    scorer.evaluate()
    scorer.accumulate()
    scorer.summarize()
    return dict(zip(names, map(float, scorer.stats[:len(names)])))
