# Setup log

Setup carried out Fri 2026-11-20, ~23:10–23:25 IST. Scope: environment, folder skeleton and docs only. No code, no datasets, no git.

## 1. Machine

| Item | Value |
|---|---|
| OS | Ubuntu (native), kernel 7.0.0-43-generic |
| CPU | 12th Gen Intel Core i5-12450H, 12 threads |
| RAM | 14 GiB usable (≈6.7 GiB available during setup) + 4 GiB swap |
| GPU | NVIDIA GeForce RTX 3050 **Ti** Laptop GPU, 4096 MiB (3.68 GiB visible to torch), compute capability 8.6 (Ampere) |
| Driver | 595.99.02, max CUDA 13.2 |
| System CUDA toolkit / cuDNN / TensorRT | None installed system-wide (no `nvcc`, no `libnvinfer`). Not needed: CUDA libraries come from pip wheels |
| Python | 3.14.4 (`/usr/bin/python3`, the only system Python); `uv` available in `~/.local/bin` |
| Disk | 49 GB free of 108 GB on `/` |
| ROS | ROS 2 "lyrical" is sourced in the shell, so `PYTHONPATH` points at `/opt/ros/lyrical/...` (see issue 1) |

### GPU check output

```
$ nvidia-smi
| NVIDIA-SMI 595.99.02              Driver Version: 595.99.02      CUDA Version: 13.2     |
|   0  NVIDIA GeForce RTX 3050 ...    Off |   00000000:01:00.0 Off |                  N/A |
| N/A   47C    P0             11W /   75W |      12MiB /   4096MiB |      7%      Default |

$ nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
NVIDIA GeForce RTX 3050 Ti Laptop GPU, 595.99.02, 4096 MiB
```

```
torch 2.15.0+cu132 cuda build 13.2 cudnn 92600
cuda available True
NVIDIA GeForce RTX 3050 Ti Laptop GPU (8, 6) 3.68 GiB
matmul ok True
torchvision 0.29.2+cu132
onnxruntime 1.32.0 ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
openvino 2026.4.1 ['CPU', 'GPU']        # 'GPU' here = Intel iGPU, not the NVIDIA card
cv2 5.1.0
ultralytics 8.4.216
```

## 2. Commands run (in order)

```bash
# --- checks ---
nvidia-smi
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
python3 --version                       # Python 3.14.4
df -h .                                 # 49G free
nproc; free -h; lscpu
ls /usr/bin/python3*; which uv          # only 3.14; uv present

# --- dry-run resolve (scratch dir, nothing installed) to confirm cp314 wheels exist ---
uv pip compile req.in --python-version 3.14 --python-platform x86_64-manylinux_2_28 --only-binary :all:
#   -> all 13 requested packages resolve; torch==2.15.0 (CUDA 13.2 build), onnxruntime-gpu==1.32.0

# --- venv + install ---
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
uv pip install --python .venv/bin/python \
    ultralytics onnx onnxslim onnxruntime-gpu openvino opencv-python numpy pandas \
    matplotlib pyyaml tqdm roboflow pycocotools
#   onnxruntime-gpu installed fine, so the onnxruntime (CPU) fallback wasn't needed

# --- verify ---
env -u PYTHONPATH .venv/bin/python -m pip check                     # No broken requirements found.
env -u PYTHONPATH .venv/bin/python -c "import torch; print(torch.cuda.is_available())"   # True

# --- smoke test (outputs written to a scratch dir, not the repo) ---
env -u PYTHONPATH .venv/bin/yolo predict model=yolo11n.pt source=<ultralytics>/assets/bus.jpg device=0   imgsz=640
env -u PYTHONPATH .venv/bin/yolo predict model=yolo11n.pt source=<ultralytics>/assets/bus.jpg device=cpu imgsz=640

# --- ORT CUDA provider runtime check (1-node Relu ONNX model in scratch dir) ---
#   -> failed: libcurand.so.10 missing (issue 3)
uv pip install --python .venv/bin/python nvidia-curand nncf
#   -> ORT CUDA provider now works; nncf installed for OpenVINO INT8 (issue 4)

# --- re-verify after nncf changed numpy ---
env -u PYTHONPATH .venv/bin/python -m pip check                     # No broken requirements found.
env -u PYTHONPATH .venv/bin/yolo predict ... device=0 / device=cpu  # both pass again

# --- TensorRT feasibility (dry run only) ---
ldconfig -p | grep -iE "nvinfer|cudnn"                              # nothing
uv pip compile <(echo tensorrt) --python-version 3.14 ...           # no cp314 wheels

# --- pin ---
env -u PYTHONPATH .venv/bin/python -m pip freeze --exclude pip      # -> requirements.txt (98 pins, grouped)
uv pip install --python .venv/bin/python -r requirements.txt --dry-run   # Would make no changes

# --- skeleton ---
for d in configs scripts src results notebooks docs data; do mkdir -p $d && touch $d/.gitkeep; done
# .gitignore written (see repo root)
```

## 3. Smoke test results

YOLO11n (COCO-pretrained, 2.6 M params, 6.5 GFLOPs), `bus.jpg`, imgsz 640 → letterboxed to 640×480, batch 1. Ultralytics runs a warm-up pass first.

| Device | Pre-process | **Inference** | Post-process | Detections |
|---|---|---|---|---|
| GPU, RTX 3050 Ti (`device=0`) | 23.8 ms | **4.4 ms** | 15.6 ms | 4 persons, 1 bus |
| CPU, i5-12450H (`device=cpu`) | 1.7 ms | **60.1 ms** | 4.4 ms | 4 persons, 1 bus |
| GPU, re-run after nncf/numpy change | 26.1 ms | **3.9 ms** | 16.2 ms | 4 persons, 1 bus |
| CPU, re-run after nncf/numpy change | 1.7 ms | **54.0 ms** | 1.5 ms | 4 persons, 1 bus |

Inference is roughly 12–15× faster on the GPU (≈ 230–255 FPS model-only on GPU vs ≈ 17–19 FPS on CPU). These are **single-image** numbers. The large GPU pre/post times are first-call overhead (CUDA context, host↔device copies), and a single run is not a benchmark. Real numbers will come from `scripts/benchmark.py` (200 images, 20 warm-up, mean + p95; PLAN.md §4).

`yolo11n.pt` (5.4 MB) was downloaded into the repo root by Ultralytics. It's git-ignored by `*.pt`.

## 4. Issues hit and fixes

| # | Issue | Cause | Fix / workaround |
|---|---|---|---|
| 1 | `pip check` reported `launch-ros requires osrf-pycommon`, `generate-parameter-library-py requires typeguard` | The sourced ROS 2 environment puts `/opt/ros/lyrical/lib/python3.14/site-packages` (+ Gazebo paths) on `PYTHONPATH`, which leaks into the venv | Run project Python with `env -u PYTHONPATH .venv/bin/python …` (or `unset PYTHONPATH` in the terminal you use for this project). After that: *No broken requirements found*. Shell config was not changed |
| 2 | `FileNotFoundError: '<stdin>'` from `multiprocessing/forkserver.py` during an inline import check | Python 3.14 defaults to the `forkserver` start method; a child process tried to re-import a `python - <<EOF` heredoc as `__main__` | Not a package problem. Real `.py` files are fine. Keep `if __name__ == "__main__":` guards in scripts (also needed for DataLoader workers) |
| 3 | ORT `CUDAExecutionProvider` silently fell back to CPU: `libcurand.so.10: cannot open shared object file` | torch's CUDA 13 wheels don't include cuRAND, but ORT's CUDA provider links against it | `uv pip install nvidia-curand` (10.4.5.25, in venv). Also: **call `onnxruntime.preload_dlls()` (or `import torch`) before creating an ORT session**, otherwise the pip-wheel CUDA libs aren't found. Verified: `['CUDAExecutionProvider', 'CPUExecutionProvider']` |
| 4 | `nncf` not installed; Ultralytics' OpenVINO INT8 export would `pip install` it on the fly (unpinned) | Ultralytics treats it as a lazy export-time dependency | Installed `nncf==3.4.0` now and pinned it. Side effect: `numpy` 2.5.5 → 2.4.6, `networkx` 3.7 → 3.6.1. `pip check` clean, smoke tests re-run and pass |
| 5 | Both `opencv-python` and `opencv-python-headless` installed | `ultralytics` requires the first, `roboflow` the second; both provide `cv2` | No conflict while both are the same version (5.1.0.95, pinned). Don't upgrade one without the other |
| 6 | Ultralytics created `~/.config/Ultralytics/settings.json` on first import | Default Ultralytics behaviour (outside the repo) | Left as is. Ultralytics can also auto-install missing packages at runtime; set `YOLO_AUTOINSTALL=false` if you want that blocked during benchmarks |

## 5. Things not done (by design)

- No `.py` files, no dataset downloads, no `git init`/commit/push.
- Nothing installed system-wide; no driver or apt changes; shell config untouched.

## 6. TensorRT (not installed)

What it would need on this machine:

| Requirement | Status here |
|---|---|
| Driver | 595.99 supports CUDA 13.2, so TensorRT **10.x built for CUDA 13** is the matching line. A recent enough driver is already there |
| GPU | sm_86 (Ampere) supports FP16 and INT8 tensor cores |
| CUDA runtime / cuDNN | Already present as pip wheels (CUDA 13.2, cuDNN 9.26) |
| **Python bindings** (`tensorrt` / `tensorrt-cu13` wheels) | **Blocker: no wheels for Python 3.14 yet.** `tensorrt-cu13` only resolves to a `0.0.0a0` placeholder; `tensorrt` has no cp314 builds |
| VRAM | 4 GB is enough for a nano/small engine build. Keep the builder workspace modest (Ultralytics `workspace=2` GB or less) and close other GPU apps |

Options, in order of least disruption:

1. **Separate Python 3.12 venv just for TensorRT export and benchmark** (e.g. `uv venv --python 3.12 .venv-trt`). `uv` would download a managed Python 3.12 into `~/.local/share/uv/` (**outside the repo, so it needs your OK**). Then `pip install tensorrt-cu13` (pip wheels bundle `libnvinfer`), plus the same ultralytics/torch pins. Export `best.pt` → `.engine` there; benchmark the engine there too, since engines are tied to the TensorRT version.
2. **NVIDIA TensorRT tarball/apt (system-wide)** for the C++ libs, then point ORT's `TensorrtExecutionProvider` at them via `LD_LIBRARY_PATH`. This gets TRT through ORT without Python bindings, but it's a system install (**needs your OK**).
3. **Skip TensorRT; use ORT CUDA FP16** (convert the ONNX model to FP16 or use `export(format="onnx", half=True)`) as the GPU FP16 row. Works in the current venv today. PLAN.md §5 already lists "or ORT CUDA FP16" as acceptable.
