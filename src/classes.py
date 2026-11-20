"""The three target classes.

Everything in the project (labels on disk, the trained model, the distance
module, the plots) refers to classes by these integer ids. Keeping them in
one place means a typo in one script can't silently shift every label.
"""

from __future__ import annotations

# Order matters: index == YOLO class id.
CLASS_NAMES: list[str] = ["cone", "barrier", "stop_sign"]

# Human-readable names used on the annotated images ("Cone, 1.5m").
DISPLAY_NAMES: dict[str, str] = {
    "cone": "Cone",
    "barrier": "Barrier",
    "stop_sign": "Stop Sign",
}

NAME_TO_ID: dict[str, int] = {name: i for i, name in enumerate(CLASS_NAMES)}

# In the 80-class COCO model that ships with Ultralytics, "stop sign" is
# class index 11. (In the raw COCO JSON the *category id* is 13, because COCO
# category ids skip numbers; see src/data/coco.py.) We need this to score the
# un-fine-tuned COCO model against our 3-class labels.
COCO80_TO_TARGET: dict[int, int] = {11: NAME_TO_ID["stop_sign"]}


def display_name(class_id: int) -> str:
    """Return the label text for a class id, e.g. 2 -> 'Stop Sign'."""
    return DISPLAY_NAMES[CLASS_NAMES[class_id]]
