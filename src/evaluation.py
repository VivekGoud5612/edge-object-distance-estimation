"""One mAP evaluator for every model.

Why not just call `model.val()`?
  Ultralytics' validator assumes the model's classes ARE the dataset's
  classes. The raw COCO model has 80 classes, our data has 3, so model.val()
  can't score it. To make "raw YOLO vs fine-tuned YOLO" a fair, like-for-like
  comparison we score *both* with this same code: same images, same labels,
  same matching rule, same metric implementation (Ultralytics' ap_per_class).

Metric recap
  * A prediction is a true positive at IoU threshold t if it overlaps an
    unmatched ground-truth box of the same class with IoU >= t
    (greedy, highest confidence first, each GT matched once).
  * AP = area under the precision-recall curve for one class.
  * mAP50 = mean AP at IoU 0.5; mAP50-95 = mean over IoU 0.50, 0.55, ..., 0.95.
  * Evaluate with a LOW confidence threshold (0.001) so the whole PR curve exists.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from tqdm import tqdm

from src.classes import CLASS_NAMES
from src.data.yolo_labels import image_to_label_path, list_images, read_yolo_labels
from src.types import Detection

IOU_THRESHOLDS = np.linspace(0.5, 0.95, 10)


def box_iou_np(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between (N,4) and (M,4) xyxy boxes -> (N,M)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(br - tl, 0, None).prod(axis=2)
    area_a = (a[:, 2:] - a[:, :2]).prod(axis=1)
    area_b = (b[:, 2:] - b[:, :2]).prod(axis=1)
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def match_image(pred_boxes: np.ndarray, pred_cls: np.ndarray, pred_conf: np.ndarray,
                gt_boxes: np.ndarray, gt_cls: np.ndarray) -> np.ndarray:
    """Return a (num_pred, 10) bool array: is each prediction a TP at each IoU threshold?"""
    correct = np.zeros((len(pred_boxes), len(IOU_THRESHOLDS)), dtype=bool)
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return correct
    iou = box_iou_np(pred_boxes, gt_boxes)
    iou = iou * (pred_cls[:, None] == gt_cls[None, :])  # only same-class matches count
    order = np.argsort(-pred_conf)
    for t_idx, t in enumerate(IOU_THRESHOLDS):
        taken = np.zeros(len(gt_boxes), dtype=bool)
        for p in order:
            candidates = np.where((iou[p] >= t) & ~taken)[0]
            if len(candidates):
                g = candidates[np.argmax(iou[p, candidates])]
                taken[g] = True
                correct[p, t_idx] = True
    return correct


@dataclass
class EvalResult:
    map50: float
    map50_95: float
    per_class: dict[str, dict[str, float]]   # name -> {ap50, ap50_95, precision, recall, n_gt}
    n_images: int

    def as_row(self) -> dict[str, float]:
        """Flat dict for a results table."""
        row = {"mAP50": self.map50, "mAP50-95": self.map50_95, "images": self.n_images}
        for name, m in self.per_class.items():
            row[f"AP50_{name}"] = m["ap50"]
        return row


def evaluate(predict_fn: Callable[[np.ndarray], list[Detection]],
             images_dir: str | Path, max_images: int | None = None,
             class_names: list[str] = CLASS_NAMES, show_progress: bool = True) -> EvalResult:
    """Score any `predict_fn(image) -> list[Detection]` on a YOLO-format split.

    `images_dir` is e.g. data/merged/images/test; labels are found by the
    standard images/ -> labels/ path swap.
    """
    from ultralytics.utils.metrics import ap_per_class

    images = list_images(images_dir)
    if max_images:
        images = images[:max_images]

    stats_tp, stats_conf, stats_pcls, stats_tcls = [], [], [], []
    for img_path in tqdm(images, desc="eval", disable=not show_progress):
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        h, w = image.shape[:2]
        gt = read_yolo_labels(image_to_label_path(img_path))
        gt_cls = np.array([g.class_id for g in gt], dtype=int)
        gt_boxes = np.array([g.to_xyxy(w, h) for g in gt]).reshape(-1, 4)

        dets = predict_fn(image)
        p_boxes = np.array([[d.x1, d.y1, d.x2, d.y2] for d in dets]).reshape(-1, 4)
        p_cls = np.array([d.class_id for d in dets], dtype=int)
        p_conf = np.array([d.confidence for d in dets], dtype=float)

        stats_tp.append(match_image(p_boxes, p_cls, p_conf, gt_boxes, gt_cls))
        stats_conf.append(p_conf)
        stats_pcls.append(p_cls)
        stats_tcls.append(gt_cls)

    tp = np.concatenate(stats_tp) if stats_tp else np.zeros((0, 10), bool)
    conf = np.concatenate(stats_conf) if stats_conf else np.zeros(0)
    pcls = np.concatenate(stats_pcls) if stats_pcls else np.zeros(0, int)
    tcls = np.concatenate(stats_tcls) if stats_tcls else np.zeros(0, int)

    per_class = {name: {"ap50": 0.0, "ap50_95": 0.0, "precision": 0.0, "recall": 0.0,
                        "n_gt": int((tcls == i).sum())} for i, name in enumerate(class_names)}
    if len(tcls) == 0:
        return EvalResult(0.0, 0.0, per_class, len(images))

    out = ap_per_class(tp, conf, pcls, tcls, names=dict(enumerate(class_names)))
    p, r, ap, classes_present = out[2], out[3], out[5], out[6]
    for j, c in enumerate(classes_present):
        name = class_names[int(c)]
        per_class[name].update(ap50=float(ap[j, 0]), ap50_95=float(ap[j].mean()),
                               precision=float(p[j]), recall=float(r[j]))
    # mAP over the classes that actually appear in this split's labels.
    return EvalResult(float(ap[:, 0].mean()), float(ap.mean()), per_class, len(images))
