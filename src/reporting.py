"""Tiny helpers to write result tables as CSV + Markdown.

We write Markdown ourselves instead of pandas.to_markdown() so we don't
need the extra `tabulate` dependency.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def to_markdown(df: pd.DataFrame, floatfmt: str = ".3f") -> str:
    def fmt(v):
        if isinstance(v, float):
            return format(v, floatfmt)
        return str(v)

    header = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    body = ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([header, sep, *body]) + "\n"


def save_table(df: pd.DataFrame, out_stem: str | Path) -> tuple[Path, Path]:
    """Write <stem>.csv and <stem>.md, return both paths."""
    out_stem = Path(out_stem)
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    csv_path, md_path = out_stem.with_suffix(".csv"), out_stem.with_suffix(".md")
    df.to_csv(csv_path, index=False)
    md_path.write_text(to_markdown(df))
    return csv_path, md_path
