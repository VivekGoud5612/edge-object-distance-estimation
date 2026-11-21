"""Test every distance assumption on real data and build an error budget.

    python -m scripts.distance_analysis --weights runs/detect/yolo11n_ft/weights/best.pt
    python -m scripts.distance_analysis                      # assumptions only, no detector
    python -m scripts.distance_analysis --n-boot 500 --camera my_camera

What it produces (results/distance_analysis/)
  report.md            every test, its estimate ± bootstrap SE, and a verdict
  ground_fit_<cls>.png box height vs bottom row: should be a straight line (flat ground)
  residuals_<cls>.png  is the scatter the same everywhere, or does it bend?
  box_noise.csv/.png   detector-induced distance bias/noise per box size
  error_budget.csv/.png Monte Carlo distance error vs range, with 5-95 % band
  boxes.csv            the raw per-box data, so you can redo any analysis

See src/distance_analysis.py for the maths and why each test is valid
without ground-truth distances.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.classes import NAME_TO_ID
from src.config import REPO_ROOT, load_distance_config
from src.distance_analysis import (aspect_check, box_height_noise, collect_boxes, ground_line_fit,
                                   image_shapes, interpret_ground_fit, monte_carlo_error)
from src.reporting import save_table, to_markdown

DISTANCES = [2, 5, 10, 20, 40]


def _plots(out: Path, fits: dict, noise: pd.DataFrame, budgets: dict[str, pd.DataFrame]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for cls, fit in fits.items():
        d, e = fit["data"], fit["estimates"]
        a, y0 = e["slope_H_over_hcam"].value, e["horizon_y0_norm"].value
        fig, ax = plt.subplots(figsize=(6, 4.5))
        ax.hexbin(d.yn, d.hn, gridsize=60, bins="log", cmap="viridis", mincnt=1)
        xs = np.linspace(max(y0, d.yn.min()), d.yn.max(), 50)
        ax.plot(xs, a * (xs - y0), "r-", lw=2, label=f"h = {a:.3f}·(y − {y0:.3f})")
        ax.axvline(y0, color="r", ls=":", lw=1, label="fitted horizon")
        ax.axvline(0.5, color="w", ls="--", lw=1, label="image centre")
        ax.set_xlabel("bottom edge row / image height")
        ax.set_ylabel("box height / image height")
        ax.set_title(f"{cls}: flat-ground test ({fit['n_boxes']} boxes, {fit['n_images']} images)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out / f"ground_fit_{cls}.png", dpi=150)
        plt.close(fig)

        # Residuals binned by row: flat-ground model predicts mean 0 everywhere.
        bins = pd.qcut(d.yn, 10, duplicates="drop")
        g = d.groupby(bins, observed=True).log_ratio
        centers = [iv.mid for iv in g.mean().index]
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.errorbar(centers, g.median(), yerr=1.96 * g.std() / np.sqrt(g.count()), fmt="o-")
        ax.axhline(0, color="k", lw=1)
        ax.set_xlabel("bottom edge row / image height (deciles)")
        ax.set_ylabel("median log(h / h_line)")
        ax.set_title(f"{cls}: residuals by image row (0 = model holds)")
        fig.tight_layout()
        fig.savefig(out / f"residuals_{cls}.png", dpi=150)
        plt.close(fig)

    if not noise.empty:
        fig, ax = plt.subplots(figsize=(7, 3.8))
        for cls, g in noise.groupby("class"):
            ax.errorbar(g.box_h_px, 100 * g.dist_bias, yerr=100 * 1.96 * g.dist_bias_se.fillna(0),
                        fmt="o-", capsize=3, label=f"{cls} bias")
            ax.plot(g.box_h_px, 100 * g.dist_noise, "x--", label=f"{cls} noise (robust std)")
        ax.axhline(0, color="k", lw=1)
        ax.set_xlabel("labelled box height (px)")
        ax.set_ylabel("distance error from box (%)")
        ax.set_title("Detector-induced distance error (95% CI on bias)")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out / "box_noise.png", dpi=150)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    for name, b in budgets.items():
        ax.fill_between(b.true_m, 100 * b.p05, 100 * b.p95, alpha=0.2)
        ax.plot(b.true_m, 100 * b.median_bias, "o-", label=f"{name}: median, 5–95% band")
    ax.axhline(0, color="k", lw=1)
    ax.set_xscale("log")
    ax.set_xticks(DISTANCES)
    ax.set_xticklabels(DISTANCES)
    ax.set_xlabel("true distance (m)")
    ax.set_ylabel("distance error (%)")
    ax.set_title("Monte Carlo error budget (cone)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "error_budget.png", dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default=None, help="detector for the box-noise test (optional)")
    ap.add_argument("--data", default="data/merged")
    ap.add_argument("--fit-source", default="roadwork", help="source whose labels feed the flat-ground fit")
    ap.add_argument("--camera", default=None)
    ap.add_argument("--device", default="0")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--max-images", type=int, default=None)
    ap.add_argument("--out", default="results/distance_analysis")
    args = ap.parse_args()

    out = REPO_ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    data = REPO_ROOT / args.data
    cfg = load_distance_config(profile=args.camera)
    report: list[str] = ["# Distance assumption tests\n",
                         f"Camera profile: `{cfg.camera.name}` (f = {cfg.camera.intrinsics.fy:.0f} px at "
                         f"{cfg.camera.intrinsics.width}×{cfg.camera.intrinsics.height}). "
                         f"Bootstrap: {args.n_boot} resamples of whole images. ± = 1 standard error.\n"]

    # ---- labelled boxes from every split of the fit source (no detector needed) ----
    fit_images = sorted(p for split in ("train", "val", "test")
                        for p in (data / "images" / split).glob(f"{args.fit_source}__*"))
    print(f"[1/4] reading labels of {len(fit_images)} {args.fit_source} images")
    gt = collect_boxes(fit_images, max_images=args.max_images)
    gt.to_csv(out / "boxes_labels.csv", index=False)

    # ---- A4: image shapes / stretching ----
    report.append("## A4. Image resolution and stretching\n")
    report.append(to_markdown(image_shapes(gt)))
    asp = aspect_check(gt, n_boot=args.n_boot)
    report.append(f"\nCone box width/height: median {asp['median_w_over_h']} over {asp['n']} boxes "
                  f"(a standard cone is ≈ {asp['expected']}). Ratio observed/expected = "
                  f"{asp['implied_stretch']:.2f}. "
                  + ("No sign of anisotropic stretching." if 0.85 <= asp['implied_stretch'] <= 1.15 else
                     "Boxes are distorted relative to a real cone: the images were likely resized "
                     "non-uniformly, so pinhole distances on these images need the vertical scale fixed.")
                  + " The flat-ground test below is unaffected either way (it is a ratio of two vertical lengths).\n")

    # ---- A1 + A2: flat ground, horizon, height variability ----
    print("[2/4] flat-ground line fits")
    report.append("## A1 + A2. Flat ground, horizon and object-height variability\n")
    report.append("Model: `box_height = (H / h_cam) · (bottom_row − horizon_row)` for objects standing on "
                  "flat ground. Focal length and distance cancel, so this needs no distance labels.\n")
    fits = {}
    f_norm = cfg.camera.intrinsics.fy / cfg.camera.intrinsics.height
    fit_rows = []
    for cls in ("cone", "barrier"):
        prior = cfg.objects[NAME_TO_ID[cls]]
        try:
            fit = ground_line_fit(gt, cls, n_boot=args.n_boot)
        except ValueError as e:
            report.append(f"### {cls}\nSkipped: {e}\n")
            continue
        fits[cls] = fit
        report.append(f"### {cls} ({fit['n_boxes']} boxes, {fit['n_images']} images)\n")
        report += [f"- {line}" for line in interpret_ground_fit(fit, prior.height_m, f_norm)]
        report.append(f"\n![{cls}](ground_fit_{cls}.png) ![{cls} residuals](residuals_{cls}.png)\n")
        fit_rows.append({"class": cls, **{k: v.value for k, v in fit["estimates"].items()},
                         **{f"{k}_se": v.se for k, v in fit["estimates"].items()}})
    if fit_rows:
        save_table(pd.DataFrame(fit_rows), out / "ground_fit")

    # ---- A3: detector box-height noise ----
    noise = pd.DataFrame()
    box_err, box_h = None, None
    if args.weights:
        print("[3/4] detector box-height noise on the test split")
        from src.detector import Detector, DetectorConfig
        det = Detector(DetectorConfig(weights=args.weights, device=args.device, conf=args.conf))
        pred = collect_boxes(data / "images" / "test", det.predict, max_images=args.max_images)
        pred.to_csv(out / "boxes_test_matched.csv", index=False)
        noise = box_height_noise(pred, n_boot=args.n_boot)
        save_table(noise, out / "box_noise")
        report.append("## A3. Detector box-height noise (test split)\n")
        report.append("`dist_bias` = median of Z_from_pred_box / Z_from_label_box − 1 (how much the detector "
                      "alone shifts distance), `dist_noise` = robust std of the same. Recall = share of "
                      "labelled boxes matched at IoU ≥ 0.5.\n")
        report.append(to_markdown(noise))
        report.append("\n![box noise](box_noise.png)\n")
        cones = pred[(pred.cls == "cone") & pred.pred_h.notna()]
        box_err = (cones.pred_h / cones.h - 1).to_numpy()
        box_h = cones.h.to_numpy()
    else:
        report.append("## A3. Detector box-height noise\nSkipped (run with --weights).\n")

    # ---- Combined Monte Carlo error budget (cones) ----
    print("[4/4] Monte Carlo error budget")
    prior = cfg.objects[NAME_TO_ID["cone"]]
    f0 = cfg.camera.intrinsics.fy
    budgets = {
        "config priors": monte_carlo_error(DISTANCES, prior.height_m, prior.height_rel_uncertainty, f0,
                                           cfg.camera.focal_rel_uncertainty, box_err, box_h),
    }
    if "cone" in fits:
        sd = fits["cone"]["estimates"]["resid_robust_std"].value
        budgets["measured height spread"] = monte_carlo_error(
            DISTANCES, prior.height_m, sd, f0, cfg.camera.focal_rel_uncertainty, box_err, box_h)
        budgets["calibrated camera"] = monte_carlo_error(
            DISTANCES, prior.height_m, sd, f0, 0.02, box_err, box_h)
    report.append("## Combined error budget (cone, Monte Carlo, 20k draws per distance)\n")
    report.append("Error = estimated / true distance − 1. Sources: real cone height varies, focal length "
                  "uncertain, detector box error resampled from A3 (if run).\n")
    for name, b in budgets.items():
        report.append(f"### {name}\n")
        report.append(to_markdown(b))
        save_table(b, out / f"error_budget_{name.replace(' ', '_')}")
    report.append("\n![budget](error_budget.png)\n")
    report.append("## Not testable without measured distances\n"
                  "- **A5 focal length.** ROADWork/BDD don't publish intrinsics. Calibrate your own camera "
                  "(`scripts/calibrate_camera.py`) and photograph cones at taped distances "
                  "(`scripts/validate_distance.py`) to measure the end-to-end error directly.\n")

    _plots(out, fits, noise, budgets)
    (out / "report.md").write_text("\n".join(report) + "\n")
    print("\n".join(report))
    print(f"\nwrote {out.relative_to(REPO_ROOT)}/report.md")


if __name__ == "__main__":
    main()
