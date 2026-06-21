"""HumanEval benchmark — migrado de eval/benchmarks/humaneval.py."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from evaluation.metrics.code_metrics import evaluate_pass_at_k, pass_at_k


@dataclass
class BenchmarkResult:
    name: str
    score: float
    baseline: str
    passed: int
    total: int


DEFAULT_DATASET_PATH = Path(__file__).resolve().parents[2] / "datasets" / "humaneval" / "problems.json"


def load_humaneval_dataset(path: Path | str | None = None) -> list[dict]:
    """Carrega dataset HumanEval."""
    dataset_path = Path(path) if path else DEFAULT_DATASET_PATH
    if dataset_path.exists():
        with dataset_path.open(encoding="utf-8") as handle:
            data = json.load(handle)
        return data.get("problems", data) if isinstance(data, dict) else data
    return _default_problems()


def _default_problems() -> list[dict]:
    return [
        {
            "task_id": "HumanEval/0",
            "prompt": "def is_palindrome(s: str) -> bool:\n    \"\"\"Check if string is palindrome.\"\"\"",
            "test": "assert is_palindrome('aba') == True",
        },
        {
            "task_id": "HumanEval/1",
            "prompt": "def sum_list(nums: list[int]) -> int:\n    \"\"\"Return sum of integers.\"\"\"",
            "test": "assert sum_list([1,2,3]) == 6",
        },
    ]


class HumanEvalBenchmark:
    TARGET_SCORE = 85.2
    BASELINE = "GPT-3.5: 76.2%"

    def __init__(self, problems_path: str | Path | None = None):
        self.problems_path = problems_path

    def evaluate(
        self,
        generate_fn: Callable[[str], str],
        problems: list[dict] | None = None,
        k: int = 1,
    ) -> BenchmarkResult:
        problems = problems or load_humaneval_dataset(self.problems_path)
        passed = 0
        for problem in problems:
            output = generate_fn(problem["prompt"])
            if self._check_solution(output, problem.get("test", "")):
                passed += 1
        total = len(problems)
        score = (passed / total * 100) if total else 0.0
        return BenchmarkResult(
            name="HumanEval (Python)",
            score=round(score, 1),
            baseline=self.BASELINE,
            passed=passed,
            total=total,
        )

    def compute_pass_at_k(
        self,
        generate_fn: Callable[[str], str],
        problems: list[dict] | None = None,
        k: int = 1,
    ) -> float:
        problems = problems or load_humaneval_dataset(self.problems_path)
        return evaluate_pass_at_k(problems, generate_fn, k=k) * 100

    def _check_solution(self, output: str, test: str) -> bool:
        try:
            namespace: dict = {}
            exec(output, namespace)
            exec(test, namespace)
            return True
        except Exception:
            return False