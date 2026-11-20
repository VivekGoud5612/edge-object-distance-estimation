"""Look at the merged data before training on it.

    python -m scripts.review_data [--data data/merged] [--split train]

Prints per-class counts and box-size stats, and saves sample grids
(results/data_review/<split>_grid.jpg) so you can eyeball label quality.
Typical things to catch here: wrong class mapping (cones labelled as
barriers), boxes shifted, a source that is mostly tiny far-away objects.
"""

from __future__ import annotations

import argparse

from src.config import REPO_ROOT
from src.data.stats import box_table, sample_grid, summarize


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/merged")
    ap.add_argument("--splits", nargs="*", default=["train", "val", "test"])
    ap.add_argument("--max-images", type=int, default=None, help="limit for a quick look")
    args = ap.parse_args()

    data = REPO_ROOT / args.data
    out = REPO_ROOT / "results" / "data_review"
    out.mkdir(parents=True, exist_ok=True)
    for split in args.splits:
        img_dir = data / "images" / split
        if not img_dir.is_dir():
            continue
        df = box_table(img_dir, args.max_images)
        print(f"\n===== {split}: {df['image'].nunique() if not df.empty else 0} labelled images, "
              f"{len(df)} boxes =====")
        summary = summarize(df)
        print(summary.to_string() if not summary.empty else "(no boxes)")
        if not df.empty:
            print("\nboxes per source x class:")
            print(df.pivot_table(index="source", columns="class", values="image",
                                 aggfunc="count", fill_value=0).to_string())
            summary.to_csv(out / f"{split}_summary.csv")
        grid = sample_grid(img_dir, out / f"{split}_grid.jpg")
        print(f"sample grid -> {grid.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
