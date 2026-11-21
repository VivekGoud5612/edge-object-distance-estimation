"""Label I/O, class remapping, merging."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from src.classes import CLASS_NAMES
from src.data.merge import Sample, write_merged_dataset
from src.data.yolo_labels import (YoloBox, build_id_map, image_to_label_path, read_yolo_labels,
                                  remap_boxes, write_yolo_labels)


def test_roundtrip(tmp_path):
    p = tmp_path / "labels" / "a.txt"
    boxes = [YoloBox(0, 0.5, 0.5, 0.2, 0.4), YoloBox(2, 0.1, 0.2, 0.05, 0.05)]
    write_yolo_labels(p, boxes)
    assert read_yolo_labels(p) == pytest.approx(boxes) or read_yolo_labels(p) == boxes


def test_polygon_label_becomes_box(tmp_path):
    p = tmp_path / "x.txt"
    p.write_text("1 0.1 0.1 0.3 0.1 0.3 0.5 0.1 0.5\n")
    (b,) = read_yolo_labels(p)
    assert (b.class_id, round(b.xc, 3), round(b.yc, 3), round(b.w, 3), round(b.h, 3)) == (1, 0.2, 0.3, 0.2, 0.4)


def test_build_id_map_by_name_case_insensitive():
    src = ["Barrier", "Cone", "Drum", "Tubular Marker"]
    m = build_id_map(src, {"cone": "cone", "barrier": "barrier", "drum": None}, CLASS_NAMES)
    assert m == {0: 1, 1: 0}


def test_remap_drops_unmapped():
    boxes = [YoloBox(0, .5, .5, .1, .1), YoloBox(5, .5, .5, .1, .1)]
    assert [b.class_id for b in remap_boxes(boxes, {0: 2})] == [2]


def test_image_to_label_path():
    assert image_to_label_path("/d/images/train/x.jpg") == Path("/d/labels/train/x.txt")


def test_coco_box_conversion_clips():
    b = YoloBox.from_xywh_abs(2, -10, 0, 60, 100, 200, 100)  # starts outside the image
    assert b.w == pytest.approx(50 / 200) and b.xc == pytest.approx(25 / 200)


def test_merge_prefixes_and_drops_empty(tmp_path):
    img = tmp_path / "src" / "img.jpg"
    img.parent.mkdir()
    cv2.imwrite(str(img), np.zeros((10, 10, 3), np.uint8))
    samples = [Sample("a", "train", img, [YoloBox(0, .5, .5, .2, .2)]),
               Sample("b", "test", img, [YoloBox(1, .5, .5, .2, .2)]),
               Sample("c", "train", img, []),           # emptied by remap -> dropped
               Sample("coco_bg", "train", img, [])]     # intentional negative -> kept
    out = tmp_path / "merged"
    summary = write_merged_dataset(samples, out)
    assert (out / "images/train/a__img.jpg").exists()
    assert (out / "labels/test/b__img.txt").read_text().startswith("1 ")
    assert not (out / "images/train/c__img.jpg").exists()
    assert (out / "images/train/coco_bg__img.jpg").exists()
    assert (out / "test_b.txt").exists()
    assert summary["dropped_empty"] == 1
    assert "names" in (out / "data.yaml").read_text()


def test_original_stem_strips_roboflow_suffix():
    from src.data.merge import original_stem
    assert original_stem(Path("pittsburgh_0042_jpg.rf.3fa9c1d2.jpg")) == "pittsburgh_0042"
    assert original_stem(Path("plain_name.jpg")) == "plain_name"


def test_fill_missing_splits_moves_whole_groups():
    from src.data.merge import fill_missing_splits, group_key
    # 20 sequences x 5 frames, plus a Roboflow-augmented copy of each frame
    samples = []
    for seq in range(20):
        for frame in range(5):
            for aug in ("aaaa", "bbbb"):
                p = Path(f"seq{seq:02d}_{frame:03d}_jpg.rf.{aug}.jpg")
                samples.append(Sample("rw", "train", p, [YoloBox(0, .5, .5, .1, .1)]))
    info = fill_missing_splits(samples, 0.1, 0.1, seed=0, group_regex=r"^(seq\d+)_\d+$")
    assert info["groups"] == 20 and info["val"] == 20 and info["test"] == 20 and info["train"] == 160
    # no sequence appears in two splits
    split_of = {}
    for s in samples:
        k = group_key(s.image_path, r"^(seq\d+)_\d+$")
        assert split_of.setdefault(k, s.split) == s.split


def test_fill_missing_splits_keeps_existing_splits():
    from src.data.merge import fill_missing_splits
    samples = [Sample("x", "train", Path(f"t{i}.jpg"), []) for i in range(10)]
    samples += [Sample("x", "val", Path(f"v{i}.jpg"), []) for i in range(3)]
    info = fill_missing_splits(samples, 0.2, 0.2, seed=0)
    assert info["val"] == 3          # untouched
    assert info["test"] == 3         # round(13 * 0.2) carved from train


def test_group_block_buckets_consecutive_photo_numbers():
    from src.data.merge import group_key
    a = group_key(Path("IMG_3931_JPG_jpg.rf.ced9df65.jpg"), group_block=25)
    b = group_key(Path("IMG_3949_JPG_jpg.rf.0679beb2.jpg"), group_block=25)
    c = group_key(Path("IMG_3951_JPG_jpg.rf.f820fe18.jpg"), group_block=25)
    assert a == b != c
