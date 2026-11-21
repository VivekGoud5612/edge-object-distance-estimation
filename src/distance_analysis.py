"""Statistical tests of every assumption behind the distance estimate.

The distance formula  Z = f * H / h  is only as good as its assumptions.
Instead of asserting them, this module TESTS each one on real data and
reports a number with an error bar (bootstrap standard error, resampling
whole images so correlated boxes in one photo don't fake precision).

Assumption                         Test (function)                     Needs ground-truth distance?
---------------------------------  ----------------------------------  ----------------------------
A1 flat ground + known horizon     ground_line_fit()                   no
A2 real object height is constant  ground_line_fit() residual spread   no
A3 detector box height is unbiased box_height_noise()                  no (uses labelled boxes)
A4 images are not stretched        aspect_check(), image_shapes()      no
A5 focal length is right           calibration + validate_distance.py  yes (your own photos)
Combined                           monte_carlo_error()                 no (propagates A2, A3, A5)

The key trick for A1/A2 (no distance labels needed)
---------------------------------------------------
For an object standing on flat ground, seen by a camera at height h_cam:

    box height        h      = f * H     / Z        (pinhole)
    below horizon     y - y0 = f * h_cam / Z        (ground plane)

Divide one by the other and both f and Z cancel:

    h = (H / h_cam) * (y_bottom - y0)

So across thousands of cones, box height must be a STRAIGHT LINE in the
bottom-edge row, through the horizon y0. We fit it in log space,
log h = log a + k·log(y - y0), robustly and with size-dependent weights:
  * slope a = H / h_cam      -> implied camera height for a 0.70 m cone
  * horizon row y0           -> camera pitch (if y0 != image centre)
  * exponent k (should be 1) -> specification test: is the model right?
  * residual spread, per box size -> how much real heights (+ label noise) vary
Each camera (image resolution) must be fitted separately.
This test is independent of the focal length AND of vertical image
stretching (both h and y scale by the same factor), which matters because
we don't know the camera that took ROADWork's photos.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.classes import CLASS_NAMES
from src.data.yolo_labels import image_to_label_path, list_images, read_yolo_labels
from src.evaluation import box_iou_np
from src.types import Detection

# --------------------------------------------------------------------------
# Data collection
# --------------------------------------------------------------------------


def collect_boxes(images: str | Path | list[Path],
                  predict_fn: Callable[[np.ndarray], list[Detection]] | None = None,
                  max_images: int | None = None, iou_match: float = 0.5) -> pd.DataFrame:
    """One row per labelled box (pixels), optionally matched to a predicted box.

    Columns: image, source, img_w, img_h, cls, x1, y1, x2, y2, w, h,
             [pred_h, pred_y2, pred_conf, iou]   (NaN when the detector missed it)
    """
    paths = images if isinstance(images, list) else list_images(images)
    if max_images:
        paths = paths[:max_images]
    rows = []
    for p in tqdm(paths, desc="boxes", disable=len(paths) < 50):
        img = cv2.imread(str(p)) if predict_fn is not None else None
        if predict_fn is not None and img is None:
            continue
        if img is not None:
            H, W = img.shape[:2]
        else:
            H, W = _image_size_fast(p)
        gts = read_yolo_labels(image_to_label_path(p))
        if not gts:
            continue
        gt_xyxy = np.array([g.to_xyxy(W, H) for g in gts])
        gt_cls = np.array([g.class_id for g in gts])

        matched = [None] * len(gts)
        if predict_fn is not None:
            dets = predict_fn(img)
            if dets:
                p_xyxy = np.array([[d.x1, d.y1, d.x2, d.y2] for d in dets])
                p_cls = np.array([d.class_id for d in dets])
                iou = box_iou_np(gt_xyxy, p_xyxy) * (gt_cls[:, None] == p_cls[None, :])
                # Greedy one-to-one matching, best IoU first.
                used: set[int] = set()
                for gi, pj in sorted(((gi, pj) for gi in range(len(gts)) for pj in range(len(dets))),
                                     key=lambda t: -iou[t]):
                    if iou[gi, pj] < iou_match:
                        break
                    if matched[gi] is None and pj not in used:
                        matched[gi] = (dets[pj], float(iou[gi, pj]))
                        used.add(pj)

        for g, (x1, y1, x2, y2), m in zip(gts, gt_xyxy, matched):
            row = {"image": p.name, "source": p.name.split("__")[0], "img_w": W, "img_h": H,
                   "cls": CLASS_NAMES[g.class_id], "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                   "w": x2 - x1, "h": y2 - y1}
            if predict_fn is not None:
                d, iou_v = m if m else (None, np.nan)
                row.update(pred_h=d.height if d else np.nan, pred_y2=d.y2 if d else np.nan,
                           pred_conf=d.confidence if d else np.nan, iou=iou_v)
            rows.append(row)
    return pd.DataFrame(rows)


def _image_size_fast(path: Path) -> tuple[int, int]:
    """(height, width) from the file header only (PIL reads it lazily, no decode)."""
    from PIL import Image

    with Image.open(path) as im:
        w, h = im.size
    return h, w


# --------------------------------------------------------------------------
# Statistics helpers
# --------------------------------------------------------------------------


def robust_std(x: np.ndarray) -> float:
    """Standard deviation estimated from the MAD: ignores a few wild outliers."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return float("nan")
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def huber_fit(X: np.ndarray, y: np.ndarray, delta: float = 1.345, iters: int = 50) -> np.ndarray:
    """Robust linear regression (Huber loss via iteratively re-weighted least squares).

    Ordinary least squares is dragged around by a few bad labels (a cone on a
    truck bed, a box cut by occlusion). Huber treats small residuals like
    least squares and large ones like absolute error, so outliers count less.
    """
    w = np.ones(len(y))
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    for _ in range(iters):
        r = y - X @ beta
        s = robust_std(r) or 1e-9
        u = np.abs(r) / (delta * s)
        w = np.where(u <= 1, 1.0, 1.0 / np.maximum(u, 1e-9))
        sw = np.sqrt(w)
        new = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)[0]
        if np.allclose(new, beta, rtol=1e-8, atol=1e-10):
            break
        beta = new
    return beta


def cluster_bootstrap(df: pd.DataFrame, stat: Callable[[pd.DataFrame], dict[str, float]],
                      n_boot: int = 200, seed: int = 0, cluster: str = "image") -> pd.DataFrame:
    """Resample whole images (clusters) with replacement and recompute `stat`.

    Boxes in one photo share the camera, the road slope and the annotator,
    so they are NOT independent. Resampling images instead of boxes gives
    honest (wider) error bars. Returns one row per bootstrap replicate.
    """
    rng = np.random.default_rng(seed)
    groups = {k: g for k, g in df.groupby(cluster)}
    keys = np.array(list(groups))
    out = []
    for _ in range(n_boot):
        pick = rng.choice(keys, size=len(keys), replace=True)
        sample = pd.concat([groups[k] for k in pick], ignore_index=True)
        try:
            out.append(stat(sample))
        except Exception:
            continue
    return pd.DataFrame(out)


@dataclass
class Estimate:
    value: float
    se: float

    def ci95(self) -> tuple[float, float]:
        return self.value - 1.96 * self.se, self.value + 1.96 * self.se

    def __str__(self) -> str:
        return f"{self.value:.4g} ± {self.se:.2g}"


def _with_se(point: dict[str, float], boots: pd.DataFrame) -> dict[str, Estimate]:
    return {k: Estimate(v, float(boots[k].std()) if k in boots and len(boots) > 1 else float("nan"))
            for k, v in point.items()}


# --------------------------------------------------------------------------
# A1 + A2: flat ground, horizon, object-height variability
# --------------------------------------------------------------------------


def _wmad(r: np.ndarray, w: np.ndarray) -> float:
    """Weighted mean absolute deviation from the weighted median (robust spread)."""
    order = np.argsort(r)
    cw = np.cumsum(w[order])
    med = r[order][np.searchsorted(cw, cw[-1] / 2)]
    return float(np.sum(w * np.abs(r - med)) / np.sum(w)), float(med)


def _fit_log_line(yn: np.ndarray, hn: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    """Find (log a, y0) for  log h = log a + log(y - y0)  by 1-D search over y0.

    For a fixed horizon y0 the model is linear with slope exactly 1 in log
    space, so log a is just the (weighted) median offset. We choose the y0
    that makes the residuals tightest (coarse grid, then a fine grid).
    """
    hi = float(np.quantile(yn, 0.01)) - 0.003
    lo = max(0.0, hi - 0.6)
    log_h = np.log(hn)
    best = (np.inf, np.nan, np.nan)
    for grid in (np.linspace(lo, hi, 40), None):
        if grid is None:   # refine around the coarse optimum
            step = (hi - lo) / 39
            grid = np.linspace(max(lo, best[2] - step), min(hi, best[2] + step), 40)
        for y0 in grid:
            r = log_h - np.log(yn - y0)
            spread, med = _wmad(r, w)
            if spread < best[0]:
                best = (spread, med, y0)
    return best[1], best[2]


def ground_line_fit(df: pd.DataFrame, cls: str = "cone", min_h_px: float = 10,
                    border_px: float = 2, n_boot: int = 200) -> dict:
    """Fit  h = a * (y_bottom - y0)  on labelled boxes of one on-ground class.

    Done in LOG space:  log h = log a + log(y - y0).
      * Distance errors are multiplicative (a box 10 % too tall gives a
        distance 10 % too short at any range), so log residuals are the
        natural, symmetric scale; in linear space far/small boxes would be
        ignored and near/big ones dominate.
      * The scatter is larger for small boxes (a few pixels of label error
        is a big fraction of a 15 px box). We measure the scatter per
        box-size quartile and weight each box by 1/variance (two passes).
      * Specification test: free the exponent, log h = log a + k·log(y - y0).
        Flat ground + pinhole says k = 1 exactly; k != 1 means the model bends.

    Coordinates are normalised by image height. Pool only images from ONE
    camera: different cameras have different heights and horizons (the
    script fits each image resolution separately for this reason).
    """
    d = df[(df.cls == cls) & (df.h >= min_h_px) & (df.y2 < df.img_h - border_px)
           & (df.y1 > border_px)].copy()
    d["hn"] = d.h / d.img_h          # box height  / image height
    d["yn"] = d.y2 / d.img_h         # bottom row  / image height
    if len(d) < 30:
        raise ValueError(f"only {len(d)} usable {cls} boxes")
    d["size_q"] = pd.qcut(d.h, 4, labels=False, duplicates="drop")

    def stat(s: pd.DataFrame, with_tables: bool = False):
        yn, hn = s.yn.to_numpy(), s.hn.to_numpy()
        w = np.ones(len(s))
        for _ in range(2):   # pass 1 unweighted, pass 2 weighted by 1/var(size quartile)
            loga, y0 = _fit_log_line(yn, hn, w)
            r = np.log(hn) - loga - np.log(np.clip(yn - y0, 1e-6, None))
            sd_q = pd.Series(r).groupby(s.size_q.to_numpy()).apply(robust_std)
            w = 1.0 / np.maximum(sd_q.reindex(s.size_q.to_numpy()).to_numpy(), 1e-3) ** 2
        x = np.log(np.clip(yn - y0, 1e-6, None))
        k, _ = huber_fit(np.column_stack([x, np.ones(len(s))]), np.log(hn))
        out = {"slope_H_over_hcam": float(np.exp(loga)), "horizon_y0_norm": float(y0),
               "exponent_k": float(k), "resid_robust_std": robust_std(r)}
        if with_tables:
            return out, r, sd_q
        return out

    point, resid, sd_by_size = stat(d, with_tables=True)
    boots = cluster_bootstrap(d, stat, n_boot=n_boot)
    est = _with_se(point, boots)

    d["log_ratio"] = resid
    size_edges = d.groupby("size_q").h.agg(["min", "max", "count"])
    hetero = pd.DataFrame({"box_h_px": [f"{a:.0f}–{b:.0f}" for a, b in zip(size_edges["min"], size_edges["max"])],
                           "n": size_edges["count"].to_numpy(),
                           "log_scatter": sd_by_size.reindex(size_edges.index).to_numpy()})
    return {"n_boxes": len(d), "n_images": d.image.nunique(), "estimates": est, "data": d,
            "scatter_by_size": hetero}


def interpret_ground_fit(fit: dict, object_height_m: float, focal_px_norm: float | None = None) -> list[str]:
    """Turn the fit into plain-language findings (used in the report)."""
    e = fit["estimates"]
    a, y0, k, sd = (e["slope_H_over_hcam"], e["horizon_y0_norm"], e["exponent_k"],
                    e["resid_robust_std"])
    lines = []
    h_cam = object_height_m / a.value
    h_cam_se = h_cam * a.se / a.value
    lines.append(f"Slope H/h_cam = {a}. For a {object_height_m:.2f} m object that implies a camera "
                 f"{h_cam:.2f} ± {h_cam_se:.2f} m above the road. Only the RATIO is identifiable: "
                 "the same slope fits a lower camera with smaller objects.")
    lo, hi = y0.ci95()
    centre_in = lo <= 0.5 <= hi
    lines.append(f"Horizon at {y0.value * 100:.1f}% of image height (95% CI {lo * 100:.1f}–{hi * 100:.1f}%). "
                 + ("Consistent with a level camera (horizon at the centre row)."
                    if centre_in else "NOT at the centre row: the camera is pitched, so the "
                    "ground-plane method must use this horizon instead of cy."))
    if focal_px_norm:
        pitch = np.degrees(np.arctan((0.5 - y0.value) / focal_px_norm))
        lines.append(f"  -> implied pitch ≈ {pitch:+.1f}° (positive = tilted down) for f = "
                     f"{focal_px_norm:.2f} × image height.")
    k_lo, k_hi = k.ci95()
    lines.append(f"Exponent k = {k} (95% CI {k_lo:.3f}–{k_hi:.3f}; flat ground + pinhole predicts exactly 1). "
                 + ("Consistent with the model."
                    if k_lo <= 1 <= k_hi else
                    "Significantly different from 1, so the straight-line model is misspecified. Likely "
                    "causes: far objects are only labelled when unusually large (selection bias, pushes k "
                    "down), road slope/crests, lens distortion, or several object sizes mixed in one class."))
    lines.append(f"Scatter: robust std of log residual = {sd} (≈ ×/÷ {np.exp(sd.value):.2f} per box). "
                 "This is the combined variability of real object height, road slope and label noise, "
                 "i.e. the error floor of a single-box pinhole distance for this class.")
    return lines


# --------------------------------------------------------------------------
# A3: detector box-height noise
# --------------------------------------------------------------------------

SIZE_BINS = [0, 16, 32, 64, 128, 256, 10_000]


def box_height_noise(df: pd.DataFrame, n_boot: int = 200) -> pd.DataFrame:
    """Bias and noise of predicted vs labelled box height, per class and size bin.

    rel_err = h_pred / h_true - 1.  Distance is ∝ 1/h, so the induced distance
    error is  Z_pred / Z_true - 1 = 1 / (1 + rel_err) - 1  (≈ -rel_err).
    Also reports recall (fraction of labelled boxes the detector matched).
    """
    rows = []
    d = df.copy()
    d["size_bin"] = pd.cut(d.h, SIZE_BINS, right=False)
    d["rel_err"] = d.pred_h / d.h - 1
    d["dist_err"] = d.h / d.pred_h - 1
    for (cls, b), g in d.groupby(["cls", "size_bin"], observed=True):
        m = g.dropna(subset=["rel_err"])
        if len(g) < 5:
            continue
        stat = lambda s: {"bias": float(s.dist_err.median()), "noise": robust_std(s.dist_err)}  # noqa: E731
        point = stat(m) if len(m) >= 5 else {"bias": np.nan, "noise": np.nan}
        boots = cluster_bootstrap(m, stat, n_boot=n_boot) if len(m) >= 10 else pd.DataFrame()
        rows.append({"class": cls, "box_h_px": str(b), "n_labelled": len(g), "recall": len(m) / len(g),
                     "dist_bias": point["bias"],
                     "dist_bias_se": float(boots["bias"].std()) if len(boots) else np.nan,
                     "dist_noise": point["noise"],
                     "dist_noise_se": float(boots["noise"].std()) if len(boots) else np.nan,
                     "mean_iou": float(m.iou.mean()) if len(m) else np.nan})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# A4: image stretching
# --------------------------------------------------------------------------


def image_shapes(df: pd.DataFrame) -> pd.DataFrame:
    """Which resolutions/aspect ratios exist, per source."""
    g = df.drop_duplicates("image").groupby(["source", "img_w", "img_h"]).size()
    return g.rename("images").reset_index()


def aspect_check(df: pd.DataFrame, cls: str = "cone", expected_w_over_h: float = 0.5,
                 min_h_px: float = 20, n_boot: int = 200) -> dict:
    """Median box width/height for a class with a known shape, with bootstrap CI.

    A standard road cone's box is about 0.5 as wide as tall (base ~0.36 m,
    height ~0.70 m). If photos were squashed vertically (e.g. 4:3 resized to
    a square), boxes become relatively wider: observed/expected > 1 hints at
    a vertical squash by that factor; < 1 at a horizontal squash.
    """
    d = df[(df.cls == cls) & (df.h >= min_h_px)].copy()
    d["ar"] = d.w / d.h
    point = {"median_w_over_h": float(d.ar.median())}
    boots = cluster_bootstrap(d, lambda s: {"median_w_over_h": float(s.ar.median())}, n_boot=n_boot)
    est = _with_se(point, boots)["median_w_over_h"]
    return {"n": len(d), "median_w_over_h": est, "expected": expected_w_over_h,
            "implied_stretch": est.value / expected_w_over_h}


# --------------------------------------------------------------------------
# Combined error budget
# --------------------------------------------------------------------------


def monte_carlo_error(distances_m: list[float], object_height_m: float, height_rel_sd: float,
                      focal_px: float, focal_rel_sd: float, box_errors: np.ndarray | None,
                      box_err_heights: np.ndarray | None = None, n: int = 20_000,
                      seed: int = 0) -> pd.DataFrame:
    """Simulate the distance estimate many times and summarise its error.

    For each true distance Z:
      true object height  H_i ~ H0 * exp(N(0, height_rel_sd))   (real objects vary)
      assumed height      H0                                     (what the code uses)
      true focal length   f_i ~ f0 * exp(N(0, focal_rel_sd))    (our f is an assumption)
      true box height     h_i = f_i * H_i / Z
      detected height     h_i * (1 + e), e resampled from the MEASURED detector errors
                          of boxes of similar pixel size (empirical, not Gaussian)
      estimate            Z_hat = f0 * H0 / detected height
    Reports median bias, 5th/95th percentile and RMSE of Z_hat/Z - 1, plus the
    same in log space (log_bias, log_sd, factor_68 = exp(log_sd)).
    """
    rng = np.random.default_rng(seed)
    rows = []
    for Z in distances_m:
        H = object_height_m * np.exp(rng.normal(0, height_rel_sd, n))
        f = focal_px * np.exp(rng.normal(0, focal_rel_sd, n))
        h = f * H / Z
        if box_errors is not None and len(box_errors):
            if box_err_heights is not None:
                # Resample errors from boxes within a factor 2 of this size.
                h_nom = focal_px * object_height_m / Z
                near = (box_err_heights > h_nom / 2) & (box_err_heights < h_nom * 2)
                pool = box_errors[near] if near.sum() >= 30 else box_errors
            else:
                pool = box_errors
            e = rng.choice(pool, n, replace=True)
        else:
            e = np.zeros(n)
        h_det = np.maximum(h * (1 + e), 1e-6)
        rel = (focal_px * object_height_m / h_det) / Z - 1
        rows.append({"true_m": Z, "box_h_px": focal_px * object_height_m / Z,
                     "median_bias": float(np.median(rel)), "p05": float(np.percentile(rel, 5)),
                     "p95": float(np.percentile(rel, 95)), "rmse": float(np.sqrt(np.mean(rel ** 2))),
                     "abs_err_p95_m": float(np.percentile(np.abs(rel), 95) * Z),
                     # Log-space summary: errors are multiplicative, so the honest single
                     # number is a geometric factor. log_sd = 0.3 means "within ×/÷ 1.35
                     # about 68 % of the time"; it treats 2× too far and 2× too near alike.
                     "log_bias": float(np.median(np.log1p(rel))),
                     "log_sd": robust_std(np.log1p(rel)),
                     "factor_68": float(np.exp(robust_std(np.log1p(rel))))})
    return pd.DataFrame(rows)
