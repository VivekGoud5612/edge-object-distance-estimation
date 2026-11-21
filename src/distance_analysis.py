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
log h = log a + log(y - y0), robustly and with size-dependent weights:
  * slope a = H / h_cam      -> implied camera height for a 0.70 m cone
  * horizon row y0           -> camera pitch (if y0 != image centre)
  * median residual per range band (all ~0) -> specification test
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


def camera_group(image_name: str) -> str:
    """Best guess at which vehicle/camera took an image, from its file name.

    ROADWork stems look like  roadwork__<city>_<32-hex sequence id>_<frame>...
    (dash-cam videos, one fleet per city) or roadwork__IMG_1234... (phone
    photos). Same city ~ same fleet/mount; that is the finest grouping with
    enough boxes per group for a stable fit.
    """
    import re
    stem = Path(image_name).stem.split("__", 1)[-1]
    m = re.match(r"(.+?)_[0-9a-f]{32}", stem)
    if m:
        return m.group(1)
    return "phone" if stem.upper().startswith("IMG") else "other"


def _wmad(r: np.ndarray, w: np.ndarray) -> float:
    """Weighted mean absolute deviation from the weighted median (robust spread)."""
    order = np.argsort(r)
    cw = np.cumsum(w[order])
    med = r[order][np.searchsorted(cw, cw[-1] / 2)]
    return float(np.sum(w * np.abs(r - med)) / np.sum(w)), float(med)


def _y0_grid(yn: np.ndarray) -> tuple[float, float]:
    """Search range for the horizon.

    Top end: the 5th percentile of box bottoms, so up to 5 % of boxes (bad
    labels, objects on rising road) may sit above the horizon (they get zero
    weight) without dragging the horizon up to them.
    """
    hi = float(np.quantile(yn, 0.05)) - 0.003
    return max(0.0, hi - 0.6), hi


def _fit_log_line(yn: np.ndarray, hn: np.ndarray, w: np.ndarray,
                  ) -> tuple[float, float]:
    """Fit  log h = log a + log(y - y0)  by a 1-D search over the horizon y0.

    For a fixed y0 the slope in log space is exactly 1, so log a is the
    weighted median offset; y0 is chosen to make the residuals tightest
    (coarse grid, then a fine grid). Returns (log a, y0).
    """
    lo, hi = _y0_grid(yn)
    log_h = np.log(hn)
    best = (np.inf, np.nan, np.nan)
    for refine in (False, True):
        if refine:
            step = (hi - lo) / 39
            grid = np.linspace(max(lo, best[2] - step), min(hi, best[2] + step), 40)
        else:
            grid = np.linspace(lo, hi, 40)
        for y0 in grid:
            # A few boxes (bad labels, objects on a rising road) can sit ABOVE a
            # candidate horizon; log() of a negative gap is NaN. Give them zero
            # weight for this candidate instead of letting NaN poison the fit.
            ok = yn - y0 > 1e-4
            wk = np.where(ok, w, 0.0)
            x = np.log(np.where(ok, yn - y0, 1.0))
            r = log_h - x
            spread, med = _wmad(r, wk)
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
      * Scatter may change with range. Boxes are binned into quartiles of the
        REGRESSOR (bottom row, i.e. range), never of box height: binning on
        the outcome would cut the residual distribution and fake a small
        within-bin scatter. Each box is weighted by 1/variance of its bin.
      * Specification test: with k = 1 and its best horizon, the median
        residual must be ~0 in every range quartile. (A free exponent k was
        tried and dropped: k and the horizon are almost perfectly confounded,
        bootstrap corr ~ -1 even on perfect synthetic data, so the joint fit
        just runs the horizon to the edge of its range and inflates k.)
      * A horizon at the edge of the search range is not an estimate: the
        fit is flagged degenerate.

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
    d["row_q"] = pd.qcut(d.yn, 4, labels=False, duplicates="drop")

    def stat(s: pd.DataFrame, with_tables: bool = False):
        yn, hn, q = s.yn.to_numpy(), s.hn.to_numpy(), s.row_q.to_numpy()
        w = np.ones(len(s))
        for _ in range(2):   # pass 1 unweighted, pass 2 weighted by 1/var(row quartile)
            loga, y0 = _fit_log_line(yn, hn, w)
            r = np.log(hn) - loga - np.log(np.clip(yn - y0, 1e-6, None))
            sd_q = pd.Series(r).groupby(q).apply(robust_std)
            w = 1.0 / np.maximum(sd_q.reindex(q).to_numpy(), 1e-3) ** 2
        # Overall scatter vs. scatter around each quartile's own median: the gap is
        # systematic misfit of the flat-ground curve, not real object-height variation.
        r_centred = r - pd.Series(r).groupby(q).transform("median").to_numpy()
        out = {"slope_H_over_hcam": float(np.exp(loga)), "horizon_y0_norm": float(y0),
               "resid_robust_std": robust_std(r), "resid_std_within_rowbin": robust_std(r_centred)}
        # Second, better-identified specification test: with the k = 1 model and its
        # best horizon, the median residual must be ~0 in EVERY range quartile.
        # (Bootstrap gives each one an SE.) A U or slope pattern = model bends.
        meds = pd.Series(r).groupby(q).median()
        out.update({f"resid_median_q{int(i)}": float(v) for i, v in meds.items()})
        if with_tables:
            return out, r, sd_q, pd.Series(r).groupby(q).median()
        return out

    point, resid, sd_by_row, med_by_row = stat(d, with_tables=True)
    boots = cluster_bootstrap(d, stat, n_boot=n_boot)
    est = _with_se(point, boots)
    lo, _ = _y0_grid(d.yn.to_numpy())
    degenerate = point["horizon_y0_norm"] <= lo + 1e-3

    d["log_ratio"] = resid
    edges = d.groupby("row_q").yn.agg(["min", "max", "count"])
    hetero = pd.DataFrame({"bottom_row_norm": [f"{a:.2f}–{b:.2f}" for a, b in zip(edges["min"], edges["max"])],
                           "median_box_h_px": d.groupby("row_q").h.median().round(0).to_numpy(),
                           "n": edges["count"].to_numpy(),
                           "median_resid": med_by_row.reindex(edges.index).to_numpy(),
                           "log_scatter": sd_by_row.reindex(edges.index).to_numpy()})
    return {"n_boxes": len(d), "n_images": d.image.nunique(), "estimates": est, "data": d,
            "scatter_by_row": hetero, "degenerate": bool(degenerate)}


def interpret_ground_fit(fit: dict, object_height_m: float, focal_px_norm: float | None = None) -> list[str]:
    """Turn the fit into plain-language findings (used in the report)."""
    e = fit["estimates"]
    a, y0, sd = (e["slope_H_over_hcam"], e["horizon_y0_norm"],
                    e["resid_robust_std"])
    lines = []
    if fit.get("degenerate"):
        lines.append("**DEGENERATE FIT:** the horizon landed on the edge of its search range, so the "
                     "slope, horizon and camera height below are not estimates. Usually means the images "
                     "are not from one fixed camera (e.g. hand-held photos) or the scene is not flat "
                     "ground. Do not use this subset for calibration.")
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
    qs = sorted(kk for kk in e if kk.startswith("resid_median_q"))
    bad = [kk for kk in qs if np.isfinite(e[kk].se) and abs(e[kk].value) > 2 * e[kk].se
           and abs(e[kk].value) > 0.03]
    pattern = ", ".join(f"{e[kk].value:+.3f}±{e[kk].se:.3f}" for kk in qs)
    lines.append(f"Range-quartile test (near → far is q3 → q0): median residuals {pattern}. "
                 + ("All within 2 SE (or < 3 %) of zero: the flat-ground curve fits every range band."
                    if not bad else f"{len(bad)} of {len(qs)} bands are off by more than 2 SE and 3 %: "
                    "the flat-ground model is misspecified for this subset. A U pattern (+ − − +) "
                    "is the signature of several cameras/horizons mixed together."))
    sw = e["resid_std_within_rowbin"]
    lines.append(f"Scatter: robust std of log residual = {sd} (≈ ×/÷ {np.exp(sd.value):.2f} per box). "
                 "This is the combined variability of real object height, road slope and label noise, "
                 "i.e. the error floor of a single-box pinhole distance for this class. "
                 f"Around each range-quartile's own median it is {sw.value:.3f}: the difference is "
                 "systematic misfit of the flat-ground curve, which a better camera model would remove, so the "
                 "irreducible height spread is closer to the smaller number.")
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
