import numpy as np
from typing import List, Tuple, Dict, Optional
import logging

logger = logging.getLogger(__name__)


def compute_iou(box1: List[float], box2: List[float]) -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


def character_accuracy(pred: str, gt: str) -> float:
    if not gt:
        return 1.0 if not pred else 0.0
    if not pred:
        return 0.0

    pred_clean = pred.upper().replace("-", "").replace(" ", "")
    gt_clean = gt.upper().replace("-", "").replace(" ", "")

    correct = sum(p == g for p, g in zip(pred_clean, gt_clean))
    max_len = max(len(pred_clean), len(gt_clean))
    return correct / max_len if max_len > 0 else 0.0


def edit_distance(s1: str, s2: str) -> int:
    """Levenshtein edit distance between two strings."""
    s1 = s1.upper().replace("-", "").replace(" ", "")
    s2 = s2.upper().replace("-", "").replace(" ", "")

    if not s1:
        return len(s2)
    if not s2:
        return len(s1)

    dp = [[0] * (len(s2) + 1) for _ in range(len(s1) + 1)]

    for i in range(len(s1) + 1):
        dp[i][0] = i
    for j in range(len(s2) + 1):
        dp[0][j] = j

    for i in range(1, len(s1) + 1):
        for j in range(1, len(s2) + 1):
            cost = 0 if s1[i-1] == s2[j-1] else 1
            dp[i][j] = min(dp[i-1][j] + 1, dp[i][j-1] + 1, dp[i-1][j-1] + cost)

    return dp[len(s1)][len(s2)]


def plate_accuracy(predictions: List[str], ground_truths: List[str]) -> Dict[str, float]:
    """
    Compute full set of plate recognition accuracy metrics.
    """
    if not predictions or not ground_truths:
        return {}

    n = min(len(predictions), len(ground_truths))
    predictions = predictions[:n]
    ground_truths = ground_truths[:n]

    exact_matches = 0
    char_accuracies = []
    edit_distances = []

    for pred, gt in zip(predictions, ground_truths):
        pred_c = pred.upper().replace("-", "").replace(" ", "")
        gt_c = gt.upper().replace("-", "").replace(" ", "")

        if pred_c == gt_c:
            exact_matches += 1

        char_accuracies.append(character_accuracy(pred, gt))
        edit_distances.append(edit_distance(pred, gt))

    metrics = {
        "exact_match_rate": exact_matches / n,
        "mean_char_accuracy": float(np.mean(char_accuracies)),
        "mean_edit_distance": float(np.mean(edit_distances)),
        "total_samples": n,
        "exact_matches": exact_matches,
    }

    return metrics


def detection_metrics(
    pred_boxes: List[List[float]],
    gt_boxes: List[List[float]],
    iou_threshold: float = 0.5,
    pred_scores: Optional[List[float]] = None
) -> Dict[str, float]:
    """
    Compute detection metrics: precision, recall, F1, mAP.
    """
    if not pred_boxes and not gt_boxes:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "map50": 1.0}

    if not pred_boxes:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "map50": 0.0}

    if not gt_boxes:
        return {"precision": 0.0, "recall": 1.0, "f1": 0.0, "map50": 0.0}

    # Sort by score if available
    if pred_scores:
        sorted_idx = sorted(range(len(pred_scores)), key=lambda i: pred_scores[i], reverse=True)
        pred_boxes = [pred_boxes[i] for i in sorted_idx]

    matched_gt = set()
    tp = 0
    fp = 0

    for pred_box in pred_boxes:
        best_iou = 0
        best_gt_idx = -1

        for gt_idx, gt_box in enumerate(gt_boxes):
            if gt_idx in matched_gt:
                continue
            iou = compute_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_iou >= iou_threshold and best_gt_idx >= 0:
            tp += 1
            matched_gt.add(best_gt_idx)
        else:
            fp += 1

    fn = len(gt_boxes) - len(matched_gt)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "map50": precision,  # Simplified mAP at IoU=0.5
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
    }


def end_to_end_metrics(
    predictions: List[dict],
    ground_truths: List[dict],
    iou_threshold: float = 0.5
) -> Dict[str, float]:
    """
    Compute end-to-end pipeline metrics combining detection + OCR.
    Each item: {"bbox": [...], "plate_text": "..."}
    """
    detection_preds = [p.get("bbox", []) for p in predictions]
    detection_gts = [g.get("bbox", []) for g in ground_truths]

    ocr_preds = [p.get("plate_text", "") for p in predictions]
    ocr_gts = [g.get("plate_text", "") for g in ground_truths]

    det_metrics = detection_metrics(detection_preds, detection_gts, iou_threshold)
    ocr_metrics = plate_accuracy(ocr_preds, ocr_gts)

    return {
        "detection": det_metrics,
        "ocr": ocr_metrics,
        "pipeline_accuracy": det_metrics.get("recall", 0) * ocr_metrics.get("exact_match_rate", 0)
    }


class MetricsTracker:
    """
    Tracks and accumulates metrics over multiple evaluations.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.predictions = []
        self.ground_truths = []
        self.confidences = []
        self.processing_times = []

    def update(self, pred: str, gt: str, confidence: float = 0.0, time_ms: float = 0.0):
        self.predictions.append(pred)
        self.ground_truths.append(gt)
        self.confidences.append(confidence)
        self.processing_times.append(time_ms)

    def compute(self) -> Dict[str, float]:
        metrics = plate_accuracy(self.predictions, self.ground_truths)
        if self.confidences:
            metrics["mean_confidence"] = float(np.mean(self.confidences))
        if self.processing_times:
            metrics["mean_processing_time_ms"] = float(np.mean(self.processing_times))
            metrics["fps"] = 1000.0 / metrics["mean_processing_time_ms"] if metrics["mean_processing_time_ms"] > 0 else 0
        return metrics

    def summary(self) -> str:
        m = self.compute()
        return (
            f"Samples: {m.get('total_samples', 0)} | "
            f"Exact Match: {m.get('exact_match_rate', 0):.1%} | "
            f"Char Accuracy: {m.get('mean_char_accuracy', 0):.1%} | "
            f"Avg Conf: {m.get('mean_confidence', 0):.2f} | "
            f"Speed: {m.get('fps', 0):.1f} FPS"
        )
