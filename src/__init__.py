"""Edge object detection + geometric distance estimation.

Package layout (see docs/ARCHITECTURE.md for the full picture):

    src/classes.py      the 3 target classes, shared by every stage
    src/config.py       YAML loading + typed camera/distance config
    src/types.py        small dataclasses passed between stages (Detection, ...)
    src/detector.py     one predict() API over .pt / ONNX / OpenVINO / TensorRT
    src/distance.py     pinhole + ground-plane distance (pure geometry, no ML)
    src/visualize.py    draws "Cone, 1.5m" style labels
    src/bev.py          optional bird's-eye view
    src/evaluation.py   one mAP evaluator used for raw AND fine-tuned models
    src/timing.py       latency / FPS measurement helpers
    src/data/           download, remap, merge, split
"""

__version__ = "0.1.0"
