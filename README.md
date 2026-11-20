# Edge Object Distance Estimation

This project detects **cones, barriers and stop signs** in camera frames, estimates **how far away each one is** using camera geometry only, and makes the model **fast enough for edge hardware**. It was built as the ERIC Robotics take-home assignment.

```
camera frame ─► YOLO11n (fine-tuned, 3 classes) ─► pinhole / ground-plane geometry ─► "Cone, 1.5m"
```

> **Status:** the code is complete and tested on synthetic data. The result tables below are placeholders (`TBD`) until training and benchmarking run on the target laptop.

---

## 1. Problem and approach

A robot with a single camera needs to know *what* obstacles are ahead and *how far* they are, in real time, on limited hardware. The solution has three parts:

1. **Detection.** A COCO-pretrained YOLO11n is fine-tuned (transfer learning) to exactly 3 classes on public data.
2. **Distance.** No depth sensor and no learned depth are used. The distance comes from the pinhole camera model, the known real size of each object and the camera's focal length. A second, independent method uses ground-plane geometry as a cross-check (see [docs/GEOMETRY.md](docs/GEOMETRY.md)).
3. **Edge.** The same model is exported to ONNX, ONNX FP16, OpenVINO FP32 and OpenVINO INT8 at several input sizes, then timed on CPU and GPU with one benchmark script.

## 2. Results

Hardware: Intel i5-12450H (CPU), RTX 3050 Ti Laptop 4 GB (GPU), Ubuntu, Python 3.14.

### 2.1 Detection: raw vs fine-tuned (`scripts/evaluate.py`)

| Model | Test set | mAP50 | mAP50-95 | AP50 cone | AP50 barrier | AP50 stop sign |
|---|---|---|---|---|---|---|
| ROADWork Mask R-CNN (published, 15 classes, context only) | ROADWork | 0.483 | 0.299 | – | – | – |
| YOLO11n raw (COCO, not fine-tuned) | merged test | TBD | TBD | 0 (class unknown) | 0 (class unknown) | TBD |
| YOLO11n fine-tuned | merged test | TBD | TBD | TBD | TBD | TBD |
| YOLO11s fine-tuned | merged test | TBD | TBD | TBD | TBD | TBD |

### 2.2 Speed vs accuracy (`scripts/benchmark.py`)

| Model | Format | Precision | imgsz | Size MB | mAP50-95 | CPU ms / FPS | GPU ms / FPS |
|---|---|---|---|---|---|---|---|
| YOLO11n | PyTorch | FP32 | 640 | TBD | TBD | TBD | TBD |
| YOLO11n | ONNX | FP32 | 640 | TBD | TBD | TBD | TBD |
| YOLO11n | ONNX | FP16 | 640 | TBD | TBD | – | TBD |
| YOLO11n | OpenVINO | FP32 | 640 | TBD | TBD | TBD | – |
| YOLO11n | OpenVINO | INT8 | 640 | TBD | TBD | TBD | – |
| YOLO11n | OpenVINO | INT8 | 416 | TBD | TBD | TBD | – |

Batch 1, 200 test images, 20 warm-up runs, end-to-end latency (pre-processing, inference, NMS). See `results/benchmark_tradeoff.png`.

### 2.3 Distance (`scripts/validate_distance.py`)

| Method | MAE (m) | Mean relative error | Bias |
|---|---|---|---|
| Pinhole (known height) | TBD | TBD | TBD |
| Ground plane | TBD | TBD | TBD |

### 2.4 Sample outputs

`TBD: results/infer/*.jpg (with the bird's-eye view panel)`

## 3. Dataset

**BDD100K, the dataset suggested in the brief, does not contain the target classes.** Its 10 detection classes are pedestrian, rider, car, truck, bus, train, motorcycle, bicycle, traffic light and traffic sign. It has no cone or barrier class, and stop signs are merged into the generic "traffic sign". The brief marks the dataset as optional, so the training data is assembled from public sources instead. Each source is remapped to our 3 classes ([configs/data_sources.yaml](configs/data_sources.yaml)):

| Source | Gives | Classes kept → ours | License |
|---|---|---|---|
| [ROADWork](https://www.cs.cmu.edu/~roadwork/) (CMU, ICCV 2025), via its [Roboflow export](https://universe.roboflow.com/roadworkwithdirectionalpanelsallobjects/roadwork_vpanels_all_objects) | Cones and barriers in real US work zones, seen from a car | Cone → cone, Barrier → barrier. Dropped: Drum, Tubular Marker, Vertical Panel | CC BY 4.0 |
| [barrier_detection](https://universe.roboflow.com/bhornphawis-jungbunjerdsak-y3rsf/barrier_detection) (Roboflow) | More Jersey/concrete and water-filled barriers | concrete_barrier, water_barrier → barrier; traffic_cone → cone | CC BY 4.0 |
| [COCO 2017](https://cocodataset.org) | Stop signs (category 13), plus 300 background images | stop sign → stop_sign | CC BY 4.0 (annotations) |

**Class definitions.**
- **Cone:** a traffic cone.
- **Barrier:** a Jersey or concrete barrier, a water-filled barrier, or a barricade. Drums are excluded; including them is a one-line experiment in the config.
- **Stop sign:** the octagonal sign plate only, without the pole.

**Splits.** Each source's own train/val/test split is kept, to avoid leaking Roboflow's augmented copies into the test set. COCO train2017 is split 85/15 into our train and val, and COCO val2017 becomes our test set. `test_<source>.txt` gives a per-source test slice.

**Known label noise.** Unlabelled cones may appear in COCO images, and unlabelled stop signs in ROADWork images. Merging datasets that were labelled for different class sets always has this problem.

## 4. Distance estimation

| | Pinhole, known height (main) | Ground plane (cross-check) |
|---|---|---|
| Formula | `Z = f · H / h_box` | `Z = f · h_cam / (y_bottom − y_horizon)` |
| Needs | Real object height H | Camera height and pitch; flat ground |
| Works for | All classes | Cone and barrier (not stop signs on poles) |
| Main error source | H varies between objects (cones are 0.45 to 0.91 m) | Road slope, camera pitch |

**Assumptions** ([configs/camera.yaml](configs/camera.yaml)):
- BDD-style dashcam: f ≈ 1005 px (assumed 65° horizontal field of view at 1280 px, because BDD publishes no intrinsics), camera height 1.3 m.
- Object heights: cone 0.70 m, barrier 0.80 m, stop sign 0.75 m.
- For validation, a phone camera calibrated with a checkerboard (`scripts/calibrate_camera.py`).

**Reliability.**
- Boxes cut by the image border, tiny boxes and clamped values are flagged, and their label is shown as `Cone, ~14.0m`.
- Every estimate carries a first-order uncertainty: `ΔZ/Z = sqrt((ΔH/H)² + (Δf/f)² + (Δh/h)²)`.
- `scripts/validate_distance.py --sensitivity` prints the full error table.

## 5. Edge optimisation

| Technique | Implemented as | Expected effect |
|---|---|---|
| ONNX export | `export.py --variants onnx` | Portable graph that runs on ONNX Runtime CPU or CUDA |
| FP16 | `onnx_fp16`, `engine_fp16` | About half the size and faster on GPU tensor cores; tiny accuracy change |
| INT8 post-training quantisation | `openvino_int8` (NNCF, calibrated on our images) | About 4× smaller, faster on CPU; small mAP drop |
| Smaller input | `--imgsz 640 416 320` | Large speed-up; hurts small, distant objects most |
| Lighter or heavier backbone | YOLO11n vs YOLO11s | Trade between accuracy and speed |
| Pruning (L1 unstructured) | `scripts/prune.py` | Lower accuracy and **no speed-up** with dense kernels. Included to show why structured pruning or quantisation is the practical route |

TensorRT has no Python 3.14 build yet, so the GPU FP16 row uses ONNX Runtime CUDA (see [docs/SETUP.md](docs/SETUP.md)).

## 6. How to run

```bash
# 0. environment (see docs/SETUP.md). If ROS is sourced, prefix commands with: env -u PYTHONPATH
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
source .venv/bin/activate
export ROBOFLOW_API_KEY=...            # free account, Settings → API key

# 1. data: download, remap, merge, split; then LOOK at it
python -m scripts.prepare_data
python -m scripts.review_data          # results/data_review/*_grid.jpg

# 2. train (sanity run first, then the real one)
python -m scripts.train --model yolo11n --smoke
python -m scripts.train --model yolo11n
python -m scripts.train --model yolo11s           # optional comparison

# 3. raw vs fine-tuned
python -m scripts.evaluate --models raw=yolo11n.pt ft_n=runs/detect/yolo11n_ft/weights/best.pt

# 4. export + benchmark
python -m scripts.export --weights runs/detect/yolo11n_ft/weights/best.pt --imgsz 640 416 320
python -m scripts.benchmark

# 5. demo
python -m scripts.infer --weights runs/detect/yolo11n_ft/weights/best.pt --source path/to/images_or_video --bev

# 6. distance validation
python -m scripts.validate_distance --sensitivity
python -m scripts.calibrate_camera --images data/calib --pattern 9 6 --square-mm 25
python -m scripts.validate_distance --weights runs/detect/yolo11n_ft/weights/best.pt \
    --camera my_camera --csv data/distance_photos/ground_truth.csv

# tests, and a full dry run without downloads
python -m pytest
python -m scripts.make_synthetic_data && python -m scripts.prepare_data --config tests/data_sources_synthetic.yaml
```

## 7. Repository layout

```
configs/      camera.yaml (geometry assumptions), data_sources.yaml, train.yaml
src/          library: classes, config, types, detector, distance, visualize, bev, evaluation, timing, reporting
src/data/     yolo_labels (I/O + remap), roboflow, coco, merge, stats
scripts/      prepare_data, review_data, train, evaluate, export, benchmark, infer,
              prune, calibrate_camera, validate_distance, make_synthetic_data
tests/        unit tests + synthetic-data config
docs/         PLAN, ARCHITECTURE, MODULES (code walkthrough), GEOMETRY (derivations), PACKAGES, SETUP
results/      tables, plots, sample outputs (committed)
data/, runs/, models/   datasets, training runs, exports (git-ignored)
```

## 8. Limitations and next steps

- The distance error is dominated by the variation in real object size. Detecting the cone *type* would remove most of it, as would replacing box height with keypoints plus PnP (Dhall et al. 2019 report about 5% error with that approach).
- The ground-plane method assumes a flat road and a known camera pitch.
- Merged datasets carry the label noise described in §3. Pseudo-labelling with the fine-tuned model is the next step to fix it.
- Temporal smoothing (tracking each object and filtering its distance over frames) would stabilise the labels on video.

## 9. Credits

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) (AGPL-3.0).
- Datasets: ROADWork (Ghosh et al., ICCV 2025), Roboflow Universe contributors, COCO (Lin et al., 2014).
- Geometry references: Dhall, Dai and Van Gool, "Real-time 3D Traffic Cone Detection for Autonomous Driving" (2019).
- Code: MIT, see [LICENSE](LICENSE). Ultralytics is AGPL-3.0, so anyone who redistributes this code together with Ultralytics is bound by the AGPL for the combined work.
