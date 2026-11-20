# ERIC Robotics assignment: plan

**Status: LOCKED v1 (Fri Nov 20, 22:30 IST).** Changes from here are logged under "Changes" at the bottom.

## 0. Problem statement and solution

**Problem:** A robot with a single monocular camera has to know, in real time on limited hardware, *what* navigation-relevant obstacles are in front of it (cones, barriers, stop signs) and *how far away* each one is, so it can slow down, stop or route around them.

**What we deliver:** one pipeline, camera frame → detector → distance module → annotated frame (`Cone, 1.5m`), plus evidence that it runs fast enough on edge hardware.
1. **Detector:** a COCO-pretrained YOLO fine-tuned to exactly 3 classes on merged public data, since BDD100K alone lacks these classes.
2. **Distance:** no depth sensor and no learned depth. Pure geometry from known object size and camera focal length, with stated assumptions and measured error.
3. **Edge:** the same model exported and quantised (ONNX, FP16, INT8, smaller input), with a before/after table of accuracy against CPU/GPU FPS, so a reader can pick the right trade-off for a given device.

**Success looks like:** decent per-class mAP on a held-out test set, distance error measured (not just claimed), a clear speed table, and a README that explains every assumption.

Detect **cone, barrier, stop sign**, estimate distance with geometry, optimise for edge.
Deadline: Sun Nov 22 evening IST. Suggested hard code freeze: **Sun 18:00 IST**.

You write the pipeline, distance module, benchmarks and README. This file is the map, not the code.

---

## 1. Dataset review (the key finding)

**BDD100K detection has none of our three classes.** Its 10 detection classes are:
pedestrian, rider, car, truck, bus, train, motorcycle, bicycle, traffic light, traffic sign.
"traffic sign" is one generic class (stop signs are inside it, unlabelled as such). No cone, no barrier.

So the brief's "work with BDD100K" can only be partly literal. Say this openly in the README; it shows you checked the data instead of training blindly.

**Recommended data mix (merge everything into 3 classes: `0 cone`, `1 barrier`, `2 stop_sign`):**

| Source | Gives you | Effort | Notes |
|---|---|---|---|
| Roboflow Universe cone sets (search "traffic cone") | cones | Low | Export directly in YOLOv8 format. Pick 1 to 2 sets with road scenes, a few thousand images total. |
| Roboflow Universe barrier sets (search "road barrier", "construction barrier", "jersey barrier", "barricade") | barriers | Low | Weakest class: "barrier" is visually diverse. Define it in the README (e.g. Jersey/concrete, plastic water-filled, A-frame barricade). |
| COCO 2017 `stop sign` (class 11) | stop signs | Low | Ultralytics can fetch COCO; filter to images containing stop signs (~1.7k train images). |
| BDD100K images (val split or a few-thousand subset) | stop signs + realistic driving context | Medium | Pseudo-label: run a COCO-pretrained YOLO on BDD, keep `stop sign` detections that overlap a BDD `traffic sign` box, spot-check ~100. This is how BDD earns its place. |
| Mapillary Vistas v2 (optional) | cones (`object--traffic-cone`), barriers (`construction--barrier--*`), signs | High | Best quality, but registration + large download + polygon→box conversion. Only if Roboflow sets are poor. |

**Merging rules**
- One `data.yaml`, 3 names. Write a small remap script per source (source class id → 0/1/2, drop the rest).
- Drop images that end up with zero labels, except keep ~10% BDD images as background (negatives). Caveat: BDD scenes sometimes contain unlabelled cones, so don't overdo negatives.
- Watch class balance; report per-class counts in the README (a table is enough).
- Fixed split: train / val / test, with test held out until the end. Keep a small **BDD-only test slice** to show domain performance.

## 2. Model choice and fine-tuning

- **Base:** `yolo11n.pt` (Ultralytics). Fallback `yolov8n.pt` if anything breaks. Both are COCO-pretrained, so stop sign starts strong.
- **Comparison model (counts as "lighter vs heavier backbone"):** `yolo11s.pt`, same recipe.
- **Recipe (starting point):**
  - `imgsz=640`, `epochs=50` (n) / `40` (s), `batch` = largest that fits, `patience=15`, `cos_lr=True`.
  - Default optimizer (`auto`) is fine. Keep default mosaic/HSV aug; set `close_mosaic=10`.
  - Optional first try: `freeze=10` (backbone frozen) for 10 epochs, then unfrozen. Only if time; plain full fine-tune usually wins.
  - `seed=0`, `deterministic=True`, pin the Ultralytics version in `requirements.txt`.
- **Sanity first:** train 3 epochs on 10% of data to check labels/paths before the long run. Look at `train_batch*.jpg` mosaics to confirm boxes look right.
- **Improvement levers (Sun, in order):** more/cleaner barrier data → `imgsz=800` for small distant cones → `s` model → longer training. Change one thing at a time and log it.

## 3. Distance estimation (geometry)

**Main method: pinhole similar triangles**

D = f_px · H_real / h_px

- `f_px` focal length in pixels, `H_real` real object height (m), `h_px` bbox height (px).
- Use **height**, not width: height is stable as objects rotate.
- `f_px = f_mm · W_px / sensor_width_mm`, or from FOV: `f_px = (W/2) / tan(HFOV/2)`.

**BDD100K intrinsics:** there are no official per-image intrinsics (phone cameras, 1280×720). Assume HFOV ≈ 65° → `f_px ≈ 1000`. Put it in a config, state it as an assumption, and show sensitivity (±10% f ⇒ ±10% D).

**Reference heights (configurable, cite in README):**

| Class | H_real | Basis |
|---|---|---|
| Cone | 0.70 m | Common 28" road cone (small cones 0.45 m) |
| Stop sign | 0.75 m | Standard 30" octagon plate (bbox = plate only, not pole) |
| Barrier | 0.80 m | Jersey barrier ≈ 0.81 m; barricades ≈ 1.0 m. Largest error source, say so |

**Cross-check method (cheap, high value): ground-plane geometry** for objects on the road (cones, barriers):
D ≈ f_px · h_cam / (y_bottom − y_horizon), with camera height `h_cam` (≈1.3 m dashcam) and horizon row (≈ image centre row for a level camera). Report both and their agreement.

**Edge cases to handle:** boxes cut by the image border (mark distance unreliable), very small boxes (< ~10 px, noisy), clamp absurd values.

**Validation (do at least one):**
- Best: photograph a real cone/sign at tape-measured distances (2, 4, 6, 8 m) with your phone, calibrate the phone with OpenCV checkerboard (`cv2.calibrateCamera`), report error %.
- Minimum: analytical error ± from H_real and f uncertainty, plus 1 px box error at range.

**Annotation format:** `Cone, 1.5m` on the box label, exactly like the brief.

## 4. Evaluation metrics

- Detection: mAP@0.5, mAP@0.5:0.95, per-class precision/recall, confusion matrix, on held-out test and on the BDD slice.
- Distance: MAE and relative error % on your measured photos (or the sensitivity analysis).
- Speed: mean latency (ms), p95 latency, FPS, model size (MB). Batch 1, same 200 images, 20 warm-up runs, note hardware + `imgsz`.
- Qualitative: 6 to 10 annotated images + 1 short BDD video clip with distance labels.

**Comparison table (rows = models, columns = test sets):**

| Model | Merged test (3 cls) | Cone test | Barrier test | Stop sign (COCO val) | BDD slice (optional) |
|---|---|---|---|---|---|
| Dataset reference (published number, if any) | n/a | Roboflow page mAP | Roboflow page mAP | n/a | n/a |
| Raw YOLO11n (COCO, no fine-tune) | stop sign only | 0 (class unknown) | 0 (class unknown) | measured | measured |
| Fine-tuned YOLO11n | measured | measured | measured | measured | measured |
| Fine-tuned YOLO11s | measured | measured | measured | measured | measured |

Caveat for README: reference numbers come from other models and splits, so they are context, not a like-for-like score. The like-for-like comparison is raw vs fine-tuned on the same test sets.

## 5. Edge optimisation

Run every variant through the **same** benchmark script and the same val set for accuracy.

| Variant | How | Where |
|---|---|---|
| PyTorch FP32 (baseline) | `.pt` | CPU, GPU |
| ONNX FP32 | `export(format="onnx")` + ONNX Runtime | CPU, GPU |
| ONNX/TensorRT FP16 | `export(format="engine", half=True)` or ORT CUDA FP16 | GPU |
| INT8 (CPU) | `export(format="openvino", int8=True, data=...)` or ORT static quant with calibration images | CPU |
| Smaller input | `imgsz` 640 vs 416 vs 320 | both (cheap, often biggest win) |
| Lighter backbone | n vs s | both |
| Pruning (optional) | `torch.nn.utils.prune` L1 unstructured 20–30% + short fine-tune | report accuracy only; note unstructured pruning gives **no real speed-up** without sparse kernels. Honest note > fake speedup |

**Results table to fill (the core deliverable of Sunday):**

| Model | Format | Precision | imgsz | Size MB | mAP50 | mAP50-95 | CPU ms / FPS | GPU ms / FPS |
|---|---|---|---|---|---|---|---|---|

Plus one before/after image pair (FP32 vs INT8) showing the same detections.

## 6. Optional extras, ranked by value per hour

1. **Bird's-eye view via homography** (1–2 h): 4 road points → `cv2.getPerspectiveTransform`, plot detections top-down. Visual and ties back to distance.
2. **Ground-plane distance cross-check** (≈1 h): already in §3; counts as extra geometry.
3. **Tracking across frames** (1–2 h): Ultralytics `track()` (ByteTrack), smooth distance per track ID. Optical flow (`cv2.calcOpticalFlowPyrLK`) on cone boxes if you want the literal brief item.
4. **Epipolar / stereo derivation** (1 h, written only): derive Z = f·B / d in the README with a diagram. No stereo data needed.

Skip anything not started by Sun 14:00.

## 7. Repo structure

```
.
├── README.md              # results tables, images, assumptions, how to run
├── requirements.txt       # pinned
├── configs/
│   ├── data.yaml          # 3 classes, paths
│   └── camera.yaml        # f_px, h_cam, horizon, object heights
├── scripts/
│   ├── prepare_data.py    # download/remap/merge/split
│   ├── train.py
│   ├── export.py          # onnx / openvino / tensorrt
│   ├── benchmark.py       # latency/FPS + accuracy per variant
│   └── infer.py           # image/video → annotated output
├── src/
│   ├── detector.py        # wraps any backend behind one predict()
│   ├── distance.py        # pinhole + ground-plane
│   ├── visualize.py
│   └── bev.py             # optional homography
├── results/               # csv tables, plots, sample outputs
└── notebooks/             # Colab training notebook (optional)
```

Don't commit datasets or large weights; commit `best.pt` only if small (nano ≈ 6 MB is fine), or link it.

## 8. Hour-by-hour (IST)

**Fri Nov 20 night (≈21:30 – 01:00)**
- [ ] Decide compute; create repo skeleton + `requirements.txt` (30 min)
- [ ] Pick and download Roboflow cone + barrier sets, COCO stop-sign subset (60 min)
- [ ] Download a BDD100K image subset; start pseudo-labelling stop signs (30 min)
- [ ] Write remap/merge script, produce `data.yaml`, count classes (60 min)
- [ ] 3-epoch smoke train on 10% data. If it runs, start the full `yolo11n` run overnight

**Sat Nov 21 (full day)**
- [ ] 09:00 Check overnight run; inspect val predictions and confusion matrix
- [ ] 09:30–12:30 `distance.py` (pinhole + ground plane), `camera.yaml`, `infer.py` with `Cone, 1.5m` labels
- [ ] 12:30 Start `yolo11s` training in background
- [ ] 13:30–16:00 `detector.py` backend wrapper, `export.py` (ONNX first)
- [ ] 16:00–19:00 `benchmark.py`; first CPU/GPU numbers for `.pt` and ONNX
- [ ] 19:00–21:00 Distance validation photos (calibrate phone, measure)
- [ ] 21:00 Commit + push everything that works

**Sun Nov 22**
- [ ] 09:00–12:00 FP16 / INT8 exports, `imgsz` sweep, fill results table
- [ ] 12:00–14:00 One improvement pass (data fix or imgsz 800), retrain if time
- [ ] 14:00–16:00 One extra (BEV homography recommended)
- [ ] 16:00–18:00 README: dataset finding, method, assumptions, tables, sample images, limitations
- [ ] 18:00 **Code freeze**, final push, clone fresh and run `infer.py` once to verify

## 9. Risks

- Barrier data quality → define class narrowly, show per-class metrics honestly.
- Colab session drops → save to Drive, use `resume=True`.
- Unknown BDD intrinsics → state assumption + validate on your own calibrated photos.
- Time → if behind on Sat evening, drop the `s` model and pruning first.

---

## Changes

- v1 locked Fri Nov 20 22:30 IST. Added §0 problem statement.
- Fri 22:55 IST: BDD100K confirmed optional; BDD pseudo-labelling now optional. Added comparison table to §4.
