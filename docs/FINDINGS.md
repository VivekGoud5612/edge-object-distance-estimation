# Findings: distance assumptions and edge optimisation (Nov 21, 2026)

Model: `yolo11n_ft` (YOLO11n fine-tuned on the merged set; test mAP50 0.576, mAP50-95 0.407).
Hardware: i5-12450H CPU, RTX 3050 Ti Laptop GPU (4 GB).
Reproduce:

```bash
python -m scripts.benchmark                       # speed + accuracy table
python -m scripts.distance_analysis --weights runs/detect/yolo11n_ft/weights/best.pt
python -m scripts.infer --weights runs/detect/yolo11n_ft/weights/best.pt \
    --source data/merged/test_roadwork.txt --bev --method fused --out results/infer_roadwork
```

## 1. Edge optimisation

All rows use batch 1 and 200 timed calls after 20 warm-up calls. Timing is end to end: pre-process, inference, NMS and copy back. mAP is on the merged test split.

| variant | device | ms (mean) | FPS | mAP50-95 | size |
|---|---|---|---|---|---|
| PyTorch fp32 640 | CPU | 32.4 | 31 | 0.407 | |
| ONNX fp32 640 | CPU | 25.7 | 39 | 0.411 | |
| OpenVINO fp32 640 | CPU | 24.1 | 41 | 0.411 | 10.7 MB |
| **OpenVINO INT8 640** | CPU | **12.0** | **84** | **0.409** | **3.3 MB** |
| OpenVINO fp32 416 | CPU | 11.1 | 90 | 0.344 | |
| OpenVINO INT8 320 | CPU | 4.1 | 246 | 0.295 | |
| PyTorch fp32 640 | GPU | 4.9 | 205 | 0.407 | |
| ONNX fp32 640 | GPU | 6.3 | 159 | 0.411 | |
| **ONNX fp16 640** | GPU | **5.1** | **197** | **0.411** | |

The full table is in `results/benchmark.md`.

- **On CPU, INT8 quantisation is the win.** Speed goes up 2.7× over PyTorch and 2× over OpenVINO fp32, at a cost of 0.002 mAP50-95. The file is a third of the size.
- **Do not trade resolution for speed.** At 416 or 320 the model loses 0.07 to 0.11 mAP50-95. Small, far cones vanish: recall drops to 41% for boxes of 16–32 px and to 4% under 16 px (table A3 in the analysis report). INT8 at 640 beats every fp32 model at 416 on speed *and* accuracy.
- **On the GPU, YOLO11n is overhead-bound.** It takes about 4–6 ms whatever the resolution. fp16 is worth 1.24× over ONNX fp32.
- **Known issue (fixed).** OpenVINO's NPU plugin segfaulted when a second model was loaded in the same process (inside `zeInitDrivers`). The benchmark now runs each OpenVINO model in its own process.

## 2. Distance: what the label data says about our assumptions

The single-box pinhole distance is `Z = f·H/h`. It rests on four assumptions:

- **A1:** objects stand on a flat ground plane, seen from a fixed camera.
- **A2:** the object height H is known.
- **A3:** the detector box height h is accurate.
- **A4:** the image is not stretched.

ROADWork has no distance labels. We can still test A1–A4 *without* distances, from labelled boxes alone. On flat ground,

    h = (H / h_cam) · (y_bottom − y_horizon)

This holds independently of the focal length, so box height must be a straight line in the bottom row. We fit it in log space, because errors are multiplicative. The bootstrap resamples whole images, because boxes in one photo share a camera.

| test | result | verdict |
|---|---|---|
| A4 stretch | cone w/h = 0.50 (a real cone ≈ 0.5) | **holds** |
| A3 detector, boxes ≥ 32 px | distance bias +0.8 % (SE 0.2–0.6 %), noise 1.5–7.5 % | **holds**; a negligible error source |
| A3 detector, 16–32 px | bias −4.6 % ± 1.0 %, noise 11 %, recall 41 % | marginal; usable cone range ≈ 20 m at f = 1005 px |
| A1 one camera | two resolutions are two very different cameras. The 4032×3024 set is phone photos, hand-held, with scatter ×/÷ 2.2 | phone subset **unusable** for calibration |
| A1 one camera, 1080p | 18 city dash-cam fleets. Per-city horizon runs from 44.5 % to 64.1 % of image height (SE ≤ 2 points) | **fails**: one camera model cannot describe the set |
| A1 flat ground, per range band | 1080p pooled: median residual + − − + (−0.155 ± 0.021, +0.156 ± 0.022). This U shape is the signature of mixed cameras | **fails** pooled. Passes in some cities (Chicago, Detroit: 0 bands off) |
| A2 height spread | robust log scatter 0.46 pooled 1080p. Per city 0.38 on average; the best cities 0.15–0.23 | the dominant error |

What this means:

1. **The dataset is not one camera, so it cannot calibrate *our* camera.** The ratio H/h_cam and the horizon must come from the deployment camera itself: a one-off calibration, or a fit on a short clip from that robot. `distance_analysis` already does that fit.
2. **Object height is the error floor, not the detector.** For 1080p cones the irreducible spread is about ×/÷ 1.2 (the best single fleet) to ×/÷ 1.45 (the average city). Calibrating the focal length barely matters: the Monte Carlo factor_68 moves only from 1.61 to 1.55. Real gains need a height prior per object subtype (cone 0.45 vs 0.70 m, drum, barricade) or ground-plane fusion.
3. **Only the ratio H/h_cam is identifiable.** A 0.70 m cone implies a 2.5 m camera, while a 0.45 m cone implies 1.6 m. The data cannot tell these apart.
4. **The demo confirms this.** The pinhole and ground-plane estimates disagree by a median factor of 2.0 (2.5 on the phone images), so only 19 % of detections pass the fusion agreement check. The config's BDD100K camera (h_cam 1.3 m, horizon at the centre, cone 0.70 m) does not describe ROADWork's cameras. The `rel_uncertainty` it prints (0.27) is the config prior, about 2× too optimistic against the measured spread.
5. **Statistics notes for the review:**
   - A free power-law exponent (`h ∝ (y − y0)^k`) was tried and dropped. k and the horizon are almost perfectly confounded: bootstrap corr ≈ −1 even on perfect synthetic data. On real data the joint fit ran the horizon to 0 % and inflated k to 2–4.5.
   - The range-band residual test is the identifiable specification test.
   - Scatter is heteroscedastic: about 0.49 for far boxes vs 0.34 for near ones. The fit therefore weights each box by 1/variance of its range quartile. The bins are on the regressor; binning on box height would cut the residuals and fake a small scatter.
   - The fitted 1080p horizon moved from 44 % to 50 % when the weighting changed. That sensitivity is itself a symptom of the mixed cameras.

## 3. Suggested next steps (for discussion)

- Ship OpenVINO INT8 at 640 for CPU robots and ONNX fp16 for GPU.
- Add a `roadwork_1080p` camera profile only for the demo, and state that it is a pooled fit. For a real robot, use `scripts/calibrate_camera.py` together with `scripts/validate_distance.py` (cones at taped distances). That is the only direct measurement of the end-to-end error.
- Report each distance with the measured ×/÷ factor, not the config prior.
- Improve barrier data: drop `barrier_extra`, add ROADWork-style drums and barricades.
