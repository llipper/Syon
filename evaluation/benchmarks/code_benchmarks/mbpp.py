"""MBPP benchmark — migrado de eval/benchmarks/mbpp.py."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass
class BenchmarkResult:
    name: str
    score: float
    baseline: str
    passed: int
    total: int


class MBPPBenchmark:
    TARGET_SCORE = 81.3
    BASELINE = "GPT-3.5: 72.5%"

    def evaluate(
        self,
        generate_fn: Callable[[str, str], str | bool],
        tasks: list[dict] | None = None,
    ) -> BenchmarkResult:
        tasks = tasks or [
            {"prompt": "Reverse a string", "language": "python", "check": "reverse"},
            {"prompt": "Check prime number", "language": "python", "check": "prime"},
            {"prompt": "Find maximum in list", "language": "python", "check": "max"},
        ]
        passed = 0
        for task in tasks:
            result = generate_fn(task["prompt"], task["language"])
            if isinstance(result, bool):
                passed += int(result)
            elif result:
                passed += 1
        total = len(tasks)
        score = (passed / total * 100) if total else 0.0
        return BenchmarkResult(
            name="MBPP (Multi-Language)",
            score=round(score, 1),
            baseline=self.BASELINE,
            passed=passed,
            total=total,
        )