"""One detector API for every model format.

Why a wrapper?
  The benchmark must time *exactly* the code path the demo uses, otherwise
  the FPS table describes a different program. So both scripts/infer.py and
  scripts/benchmark.py go through Detector.predict().

How it works
  Ultralytics' YOLO() class already picks the right runtime from the file
  type (AutoBackend):
      *.pt                 -> PyTorch
      *.onnx               -> ONNX Runtime (CUDA provider on GPU, CPU otherwise)
      *_openvino_model/    -> OpenVINO (CPU)
      *.engine             -> TensorRT (GPU)
  We add three things on top:
    1. a stable output type (list[Detection]) the rest of the code depends on,
    2. optional class remapping, so the raw 80-class COCO model can be scored
       against our 3 classes (COCO "stop sign" 11 -> our 2, everything else dropped),
    3. the per-stage timings Ultralytics measures (pre-process, inference, NMS).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.types import Detection


@dataclass
class DetectorConfig:
    weights: str                      # path to .pt / .onnx / _openvino_model dir / .engine
    device: str = "cpu"               # "cpu" or "0" (first CUDA GPU)
    imgsz: int = 640                  # network input size (letterboxed square)
    conf: float = 0.25                # confidence threshold
    iou: float = 0.7                  # NMS IoU threshold
    half: bool = False                # FP16 compute for .pt on GPU (exported files carry their own precision)
    class_map: dict[int, int] | None = None  # model class -> target class; unmapped are dropped
    extra: dict = field(default_factory=dict)  # passed straight to YOLO.predict


class Detector:
    def __init__(self, cfg: DetectorConfig):
        # Imported lazily so that light-weight modules (distance, tests) don't
        # pay the torch import cost.
        from ultralytics import YOLO

        if cfg.device != "cpu":
            # onnxruntime-gpu needs the CUDA libs that ship inside the torch
            # wheels. Importing torch first loads them; without this the CUDA
            # provider silently falls back to CPU (see docs/SETUP.md, issue 2).
            import torch  # noqa: F401

        self.cfg = cfg
        task = "detect"
        self.model = YOLO(cfg.weights, task=task)
        self.last_speed_ms: dict[str, float] = {}

    @property
    def names(self) -> dict[int, str]:
        return self.model.names

    def predict(self, image: np.ndarray) -> list[Detection]:
        """Run detection on one BGR image (as read by cv2.imread)."""
        kwargs = dict(self.cfg.extra)
        if self.cfg.half:
            # Ultralytics 8.4 replaced the old `half=True` flag with `quantize=16`.
            kwargs["quantize"] = 16
        result = self.model.predict(
            image,
            imgsz=self.cfg.imgsz,
            conf=self.cfg.conf,
            iou=self.cfg.iou,
            device=self.cfg.device,
            verbose=False,
            **kwargs,
        )[0]
        # {'preprocess': ms, 'inference': ms, 'postprocess': ms}
        self.last_speed_ms = dict(result.speed)
        return self._to_detections(result)

    def warmup(self, image: np.ndarray, runs: int = 10) -> None:
        """First calls are slow (CUDA context, graph optimisation, caches)."""
        for _ in range(runs):
            self.predict(image)

    def _to_detections(self, result) -> list[Detection]:
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy().astype(int)

        out: list[Detection] = []
        for (x1, y1, x2, y2), c, k in zip(xyxy, confs, classes):
            if self.cfg.class_map is not None:
                if k not in self.cfg.class_map:
                    continue          # e.g. COCO "car": not one of our classes
                k = self.cfg.class_map[k]
            out.append(Detection(int(k), float(c), float(x1), float(y1), float(x2), float(y2)))
        return out


def model_size_mb(weights: str | Path) -> float:
    """Size on disk; for OpenVINO this is the whole *_openvino_model folder."""
    p = Path(weights)
    if p.is_dir():
        return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1e6
    return p.stat().st_size / 1e6
