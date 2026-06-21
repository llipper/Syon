"""Métricas de latência e uso de GPU conforme README."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable


@dataclass
class LatencyResult:
    metric: str
    elapsed_ms: float
    target_ms: float
    within_target: bool


TARGETS = {
    "inference_512_tokens": 120,
    "code_generation_100_lines": 280,
    "security_analysis_1kb": 150,
    "batch_processing_32x": 3200,
}


def measure_latency(name: str, fn: Callable[[], None]) -> LatencyResult:
    start = time.perf_counter()
    fn()
    elapsed_ms = (time.perf_counter() - start) * 1000
    target = TARGETS.get(name, elapsed_ms)
    return LatencyResult(
        metric=name,
        elapsed_ms=round(elapsed_ms, 1),
        target_ms=target,
        within_target=elapsed_ms <= target,
    )