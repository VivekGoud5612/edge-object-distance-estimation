# Code walkthrough: what each file does and why

Read the files in the order of the table, about 20 minutes each. Every file starts with a docstring that explains it, so open the file next to this page.

## Reading order

| # | File | One-line purpose | Questions to be able to answer afterwards |
|---|---|---|---|
| 1 | `src/classes.py` | Defines the 3 class ids (cone 0, barrier 1, stop_sign 2) shared by every stage. | Why does COCO "stop sign" have id 13 in the JSON but 11 in the YOLO model? |
| 2 | `src/types.py` | Defines `Detection` and `DistanceEstimate`, the objects passed between stages. | Why are boxes kept in original-image pixels and not network pixels? |
| 3 | `src/config.py` + `configs/camera.yaml` | Loads every assumption (f, camera height, object heights) from YAML. | What happens to the distance if you run at a different resolution from the calibration? (`scaled_to`) |
| 4 | `src/distance.py` + `docs/GEOMETRY.md` | Computes pinhole and ground-plane distances, flags and uncertainty. | When is the ground-plane method invalid? Why does a truncated box overestimate distance? |
| 5 | `src/detector.py` | Gives one `predict()` call for .pt, ONNX, OpenVINO and TensorRT models. | Why must the benchmark time this same code? What does `class_map` do for the raw COCO model? |
| 6 | `src/visualize.py` | Draws labels like "Cone, 1.5m", with a "~" when an estimate is unreliable. | |
| 7 | `scripts/infer.py` | Runs the whole online pipeline on an image, folder, video or webcam. | What goes into `detections.csv`, and why log both distance methods? |
| 8 | `src/data/yolo_labels.py` | Reads, writes and remaps YOLO label files. | Why map classes by name and not by id? |
| 9 | `src/data/roboflow.py`, `src/data/coco.py` | Downloads the data. | How do we fetch about 1.7k COCO images without the 19 GB zip? |
| 10 | `src/data/merge.py` + `scripts/prepare_data.py` | Merges and splits the sources and writes `data.yaml`. | Why keep each source's own splits? Why the `source__` filename prefix? |
| 11 | `src/data/stats.py` + `scripts/review_data.py` | Reports class counts and box sizes and saves sample grids. | What share of boxes are under 16 px, and what does that mean for far-away distances? |
| 12 | `scripts/train.py` + `configs/train.yaml` | Fine-tunes the model (transfer learning). | What do `close_mosaic`, `cos_lr`, `amp` and `patience` each do? |
| 13 | `src/evaluation.py` + `scripts/evaluate.py` | Uses one evaluator to compare the raw and fine-tuned models. | Why not just use `model.val()`? Why `conf=0.001` for mAP? |
| 14 | `scripts/export.py` | Exports to ONNX, FP16 and INT8 and writes `models/manifest.yaml`. | What is INT8 calibration, and why does it need our images? |
| 15 | `src/timing.py` + `scripts/benchmark.py` | Measures speed and accuracy for every variant on CPU and GPU. | Why warm up first? Why report p95 as well as the mean? Why call `cuda.synchronize`? |
| 16 | `scripts/prune.py` | Runs the pruning experiment. | Why does unstructured pruning not make the model faster? |
| 17 | `src/bev.py` | Draws the bird's-eye view using a homography. | Where do the 4 point pairs come from? |
| 18 | `scripts/calibrate_camera.py`, `scripts/validate_distance.py` | Measure the real distance error. | What is the reprojection error, and what is a good value? |

## How the pieces connect

```
configs/data_sources.yaml ─► prepare_data ─► data/merged (+data.yaml) ─► review_data
                                                     │
configs/train.yaml ─────────────────────────────► train ─► runs/detect/<name>/weights/best.pt
                                                     │
                                  evaluate (raw vs fine-tuned) ◄─┤
                                                     ▼
                                   export ─► models/*.onnx, *_openvino_model + manifest.yaml
                                                     │
                                  benchmark ◄────────┘ ─► results/benchmark.{csv,md,png}

camera frame ─► Detector.predict ─► DistanceEstimator.estimate ─► draw_detections ─► "Cone, 1.5m"
                (src/detector.py)    (src/distance.py + camera.yaml)   (src/visualize.py)
```

## Design rules in this code

- **Assumptions live in configs, not in code.** To run a sensitivity analysis, edit the YAML.
- **One code path is timed and shipped.** `benchmark.py` and `infer.py` both call `Detector.predict()`.
- **One evaluator for every model.** The raw and fine-tuned comparison is fair because both are scored by the same code on the same images.
- **Each module does one thing.** The distance module doesn't know which backend produced the boxes. The detector doesn't know about distances.
- **Pure functions for the maths.** `pinhole_distance`, `ground_plane_distance` and `relative_uncertainty` are easy to unit-test and easy to explain.

## Tests

`python -m pytest`. There are 20 tests covering the distance formulas and edge cases, label I/O and remapping, merging, the evaluator (perfect predictions must score 1.0), and agreement between the bird's-eye view and the ground-plane formula.

The scripts were also run end to end on synthetic data (`scripts/make_synthetic_data.py`) on CPU: prepare, review, train, evaluate, export (ONNX and OpenVINO FP32/INT8), benchmark, infer with the bird's-eye view, prune, and the sensitivity table. On that run the custom evaluator's mAP50-95 (0.851) matched Ultralytics' own `val()` (0.848).
