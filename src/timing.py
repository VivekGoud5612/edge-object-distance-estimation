"""Latency / FPS measurement.

Rules we follow so numbers are comparable between variants:
  * batch size 1 (a robot processes one frame at a time),
  * same images, same order, for every model,
  * warm-up runs are discarded (first runs include CUDA init, graph optimisation),
  * we report mean AND p95: a robot cares about the slow frames too,
  * FPS = 1000 / mean latency (end-to-end: pre-process + inference + NMS).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class LatencyStats:
    n: int
    mean_ms: float
    median_ms: float
    p95_ms: float
    std_ms: float

    @property
    def fps(self) -> float:
        return 1000.0 / self.mean_ms if self.mean_ms > 0 else float("nan")

    @classmethod
    def from_samples(cls, samples_ms: list[float]) -> "LatencyStats":
        a = np.asarray(samples_ms, dtype=float)
        return cls(len(a), float(a.mean()), float(np.median(a)),
                   float(np.percentile(a, 95)), float(a.std()))


def time_calls(fn: Callable[[], object], runs: int, warmup: int = 0,
               sync: Callable[[], None] | None = None) -> LatencyStats:
    """Time `fn()` `runs` times after `warmup` untimed calls.

    `sync` (e.g. torch.cuda.synchronize) makes sure GPU work has actually
    finished before we stop the clock; without it GPU timings look too fast.
    """
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        if sync is not None:
            sync()
        samples.append((time.perf_counter() - t0) * 1000.0)
    return LatencyStats.from_samples(samples)
