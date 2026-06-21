"""Coleta de métricas de inferência."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class InferenceMetrics:
    """Métricas em memória para sessões de inferência."""

    inference_times_ms: list[float] = field(default_factory=list)
    token_counts: list[int] = field(default_factory=list)
    throughputs: list[float] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    _start_time: float = field(default_factory=time.time)

    def log_inference_time(self, elapsed_ms: float) -> None:
        self.inference_times_ms.append(elapsed_ms)

    def log_token_throughput(self, token_count: int, elapsed_ms: float) -> None:
        self.token_counts.append(token_count)
        if elapsed_ms > 0:
            tps = (token_count / elapsed_ms) * 1000
            self.throughputs.append(tps)

    def log_memory_usage(self) -> dict[str, Any]:
        usage: dict[str, Any] = {"process_mb": None, "gpu_mb": None}
        try:
            import psutil

            usage["process_mb"] = round(psutil.Process().memory_info().rss / (1024**2), 2)
        except ImportError:
            pass

        try:
            import torch

            if torch.cuda.is_available():
                usage["gpu_mb"] = round(torch.cuda.memory_allocated() / (1024**2), 2)
        except ImportError:
            pass
        return usage

    def log_errors(self, error: str) -> None:
        self.errors.append(error)

    def summary(self) -> dict[str, Any]:
        avg_latency = (
            sum(self.inference_times_ms) / len(self.inference_times_ms)
            if self.inference_times_ms
            else 0.0
        )
        avg_throughput = (
            sum(self.throughputs) / len(self.throughputs) if self.throughputs else 0.0
        )
        return {
            "total_requests": len(self.inference_times_ms),
            "avg_latency_ms": round(avg_latency, 2),
            "avg_throughput_tps": round(avg_throughput, 2),
            "total_errors": len(self.errors),
            "uptime_seconds": round(time.time() - self._start_time, 2),
        }