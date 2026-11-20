# Packages

Environment: `.venv/` with Python 3.14.4 on Ubuntu. NVIDIA driver 595.99.02 (supports up to CUDA 13.2), RTX 3050 Ti Laptop GPU (4 GB, compute capability 8.6, Ampere).
Exact pins for everything, including the 81 transitive packages, are in [`../requirements.txt`](../requirements.txt).

"Where it runs" means where this project's work actually executes, not just where the package could run.

## Direct packages

| Package | Version | What it is in general | What this project uses it for | Runs on |
|---|---|---|---|---|
| `ultralytics` | 8.4.216 | YOLO framework: models, training loop, validation, export, CLI (`yolo …`) | **Training** (fine-tune YOLO11n/s on 3 classes), validation (mAP, per-class P/R, confusion matrix), **export** to ONNX / OpenVINO / TensorRT, quick prediction | Both (training on GPU; predict and val on either) |
| `torch` | 2.15.0 (+cu132) | PyTorch deep-learning library; tensors and autograd on CPU or CUDA | Backend for Ultralytics **training** and the PyTorch FP32 **benchmark** baseline; optional pruning (`torch.nn.utils.prune`) | Both |
| `torchvision` | 0.29.2 (+cu132) | Vision ops and transforms for PyTorch | Used inside Ultralytics (NMS `torchvision.ops.nms`, image ops). Not called directly | Both |
| `onnx` | 1.23.2 | ONNX model format: protobuf definition, checker, graph tools | **Export**: Ultralytics writes `.onnx` with it; we use it to inspect/check graphs and as input to static INT8 quantisation if we take the ORT route | CPU (file tooling only) |
| `onnxslim` | 0.1.99 | ONNX graph simplifier (constant folding, removes redundant nodes) | **Export**: Ultralytics runs it automatically with `simplify=True` to produce a smaller, faster ONNX graph | CPU (file tooling only) |
| `onnxruntime-gpu` | 1.32.0 | Microsoft's ONNX inference engine, GPU build. Ships CPU, CUDA and TensorRT execution providers | **Benchmarking** ONNX FP32/FP16 on CPU and GPU; `onnxruntime.quantization` for optional static **INT8 quantisation** with calibration images | Both (`CPUExecutionProvider`, `CUDAExecutionProvider` verified working; `TensorrtExecutionProvider` is listed but needs TensorRT libs, not installed). **Call `onnxruntime.preload_dlls()` (or `import torch`) before creating a session**, otherwise CUDA silently falls back to CPU |
| `openvino` | 2026.4.1 | Intel's inference toolkit: model converter and runtime for Intel CPU/iGPU/NPU. INT8 calibration is done by `nncf` (below) | **Export** + **INT8 quantisation** for CPU (`export(format="openvino", int8=True, data=...)`), and CPU **benchmarking** of the OpenVINO FP32/INT8 models | CPU (also detects the Intel iGPU as `GPU`; it can't use the NVIDIA card) |
| `nncf` | 3.4.0 | Intel's Neural Network Compression Framework: post-training quantisation, pruning | **INT8 quantisation**: Ultralytics calls it inside `export(format="openvino", int8=True)` to calibrate on val images. Added during setup so it's pinned (Ultralytics would otherwise pip-install it mid-export) | CPU |
| `nvidia-curand` | 10.4.5.25 | NVIDIA cuRAND random-number library (CUDA 13), as a pip wheel | Provides `libcurand.so.10`, which `onnxruntime-gpu`'s CUDA provider needs and the torch wheels don't ship. Added during setup to fix GPU **benchmarking** of ONNX | GPU |
| `opencv-python` | 5.1.0.95 | OpenCV computer-vision library (`cv2`) with GUI support (`imshow`) | Image/video I/O, drawing `Cone, 1.5m` boxes, letterbox resize, `calibrateCamera` for distance validation, `getPerspectiveTransform` for the bird's-eye view | CPU |
| `numpy` | 2.4.6 | N-dimensional arrays and maths | Box maths, distance formulas, pre/post-processing for ONNX/OpenVINO backends, latency statistics (mean, p95) | CPU |
| `pandas` | 3.1.1 | Tabular data frames | **Benchmarking**: collect results per variant, write `results/*.csv` and Markdown tables; class-count tables in data prep | CPU |
| `matplotlib` | 3.11.2 | 2D plotting | **Plotting**: accuracy-vs-FPS trade-off charts, distance error plots, BEV plot | CPU |
| `PyYAML` | 6.0.3 | YAML parser/writer (`import yaml`) | Read/write `configs/data.yaml` and `configs/camera.yaml`; read Roboflow `data.yaml` files during **label conversion** | CPU |
| `tqdm` | 4.70.1 | Progress bars | Progress for **data download**, remap/merge loops, benchmark runs | CPU |
| `roboflow` | 1.7.3 | Python SDK for Roboflow Universe/workspaces | **Data download**: pull cone and barrier datasets in YOLOv8 format (needs a Roboflow API key) | CPU (network) |
| `pycocotools` | 2.0.11 | Official COCO API: annotation loading and COCO-style mAP evaluation | **Label conversion**: read COCO 2017 `instances_*.json`, filter `stop sign` (category 13 in COCO JSON = class 11 in the 80-class YOLO index), convert boxes to YOLO txt; optional COCO-style eval | CPU |

## Notable transitive packages

| Package | Version | Why it's here |
|---|---|---|
| `opencv-python-headless` | 5.1.0.95 | Pulled in by `roboflow`. Same `cv2` module as `opencv-python` minus GUI. Both are at the same version so they don't conflict, but if one is ever upgraded alone, `cv2` can break. Keep them pinned together |
| `nvidia-cudnn-cu13`, `nvidia-cublas`, `nvidia-cuda-runtime`, `cuda-toolkit`, … | 9.26 / 13.x | CUDA 13 runtime libraries shipped as pip wheels for torch. No system CUDA toolkit is needed. `onnxruntime-gpu`'s CUDA provider loads the same libraries via `preload_dlls()` |
| `triton` | 3.9.0 | GPU kernel compiler used by `torch.compile`. Not used directly |
| `onnxruntime` deps: `flatbuffers`, `protobuf`, `sympy`, `ml-dtypes` | see pins | ONNX / ORT internals |
| `polars`, `psutil`, `ultralytics-thop`, `ultralytics-platform` | see pins | Ultralytics internals (FLOPs counting, system info, results tables) |
| `scipy`, `scikit-learn`, `pydot`, `ninja`, `safetensors` | see pins | Pulled in by `nncf`. Note: installing `nncf` downgraded `numpy` 2.5.5 → 2.4.6 and `networkx` 3.7 → 3.6.1; smoke tests re-run and pass |
| `openvino-telemetry` | 2025.2.0 | OpenVINO usage telemetry. Opt out with `opt_in_out --opt_out` if wanted |

## Variants explained

### `onnxruntime` vs `onnxruntime-gpu`

| | `onnxruntime` | `onnxruntime-gpu` (installed) |
|---|---|---|
| Execution providers | CPU only (plus small extras like `AzureExecutionProvider`) | CPU **+ CUDA + TensorRT** |
| CPU speed | Same CPU kernels | Same CPU kernels, so CPU benchmarks are identical |
| Size | small (tens of MB) | large (hundreds of MB) |
| Needs | Nothing | CUDA 13 + cuDNN 9 runtime (here: pip wheels from torch + `nvidia-curand`, loaded with `ort.preload_dlls()`); TensorRT libs for the TRT provider |

**Install only one.** Both provide the same `onnxruntime` Python module and overwrite each other. The GPU build covers both CPU and GPU benchmarks, so it's the right choice here. No fallback was needed.

### What OpenVINO adds

- A runtime tuned for **Intel CPUs** (this laptop: i5-12450H, AVX2/VNNI). It's often faster than ONNX Runtime CPU for the same FP32 model.
- **Post-training INT8 quantisation** with a calibration set. Ultralytics handles it with `int8=True, data=configs/data.yaml`. This is the planned CPU INT8 path in PLAN.md §5.
- An "edge CPU" story: the same IR runs on Intel iGPU/NPU devices. It doesn't touch the NVIDIA GPU.

### What TensorRT would add (not installed)

- NVIDIA's graph optimiser and compiler. It fuses layers, auto-tunes kernels for this exact GPU (sm_86) and builds a `.engine` file.
- Real **FP16** (and INT8 with calibration) speed-ups on the RTX 3050 Ti. This is usually the fastest GPU path, typically 1.5 to 3× over PyTorch FP32 for a nano model.
- Two ways to use it: Ultralytics `export(format="engine", half=True)` (needs the `tensorrt` Python package), or ORT's `TensorrtExecutionProvider` (needs the TensorRT shared libraries on the library path).
- Engines are tied to the GPU model and the TensorRT version: rebuild them on the target device.

See [SETUP.md §6](SETUP.md#6-tensorrt-not-installed) for what installing it would need on this machine.
