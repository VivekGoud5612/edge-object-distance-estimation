"""Robust HTTP download: progress bar, timeout, retries, resume.

Why not urllib.request.urlretrieve?
  It has no timeout (a stalled connection hangs forever), no progress bar
  (you can't tell "slow" from "stuck"), and an interrupted download leaves
  a truncated file that looks complete on the next run.

How this works
  * Data goes to <dest>.part and is renamed to <dest> only when complete,
    so an existing <dest> always means "finished".
  * If a .part file exists, we ask the server for the rest of the file
    (HTTP Range request) instead of starting over. If the server doesn't
    support ranges (answers 200 instead of 206), we start from zero.
  * `timeout` is per network read: if no bytes arrive for that long, the
    attempt fails and is retried (with back-off), resuming where it stopped.
"""

from __future__ import annotations

import time
import urllib.error
import urllib.request
from pathlib import Path

from tqdm import tqdm

CHUNK = 1 << 20  # 1 MiB


def download_file(url: str, dest: str | Path, timeout: float = 30.0, retries: int = 5,
                  progress: bool = True) -> Path:
    """Download `url` to `dest` (resumable). Returns `dest`. Raises after `retries` failures."""
    dest = Path(dest)
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")

    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            _download_attempt(url, part, timeout, progress)
            part.rename(dest)
            return dest
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
            last_error = e
            wait = min(2 ** attempt, 30)
            if progress:
                print(f"[download] attempt {attempt}/{retries} failed ({e}); retrying in {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"could not download {url} after {retries} attempts: {last_error}")


def _download_attempt(url: str, part: Path, timeout: float, progress: bool) -> None:
    have = part.stat().st_size if part.exists() else 0
    req = urllib.request.Request(url, headers={"User-Agent": "dataset-downloader"})
    if have:
        req.add_header("Range", f"bytes={have}-")

    with urllib.request.urlopen(req, timeout=timeout) as resp:
        if have and resp.status != 206:
            have = 0  # server ignored the Range header: start over
        length = resp.headers.get("Content-Length")
        total = have + int(length) if length is not None else None
        if total is not None and have >= total:
            return
        mode = "ab" if have else "wb"
        with open(part, mode) as f, tqdm(total=total, initial=have, unit="B", unit_scale=True,
                                         unit_divisor=1024, desc=part.name.removesuffix(".part"),
                                         disable=not progress) as bar:
            while True:
                chunk = resp.read(CHUNK)   # raises TimeoutError if the socket stalls
                if not chunk:
                    break
                f.write(chunk)
                bar.update(len(chunk))

    size = part.stat().st_size
    if total is not None and size != total:
        raise ConnectionError(f"incomplete download: {size} of {total} bytes")
