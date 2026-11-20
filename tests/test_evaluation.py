"""The shared mAP evaluator: perfect predictions must score 1, empty must score 0."""

import cv2
import numpy as np

from src.data.yolo_labels import YoloBox, write_yolo_labels
from src.evaluation import box_iou_np, evaluate
from src.types import Detection


def make_split(tmp_path):
    img_dir = tmp_path / "images" / "test"
    img_dir.mkdir(parents=True)
    gts = {}
    for i in range(3):
        cv2.imwrite(str(img_dir / f"{i}.jpg"), np.zeros((100, 200, 3), np.uint8))
        boxes = [YoloBox(i % 3, 0.5, 0.5, 0.2, 0.4)]
        write_yolo_labels(tmp_path / "labels" / "test" / f"{i}.txt", boxes)
        gts[i] = boxes
    return img_dir, gts


def test_iou():
    a = np.array([[0, 0, 10, 10]], float)
    assert abs(box_iou_np(a, a)[0, 0] - 1.0) < 1e-6
    assert box_iou_np(a, np.array([[10, 10, 20, 20]], float))[0, 0] == 0.0


def test_perfect_and_empty(tmp_path):
    img_dir, _ = make_split(tmp_path)
    # all three images have a box at the centre: x 80..120, y 30..70
    calls = {"i": 0}

    def perfect(image):
        k = calls["i"] % 3
        calls["i"] += 1
        return [Detection(k, 0.9, 80, 30, 120, 70)]

    res = evaluate(perfect, img_dir, show_progress=False)
    assert abs(res.map50 - 1.0) < 0.01 and abs(res.map50_95 - 1.0) < 0.01

    res = evaluate(lambda im: [], img_dir, show_progress=False)
    assert res.map50 == 0.0
