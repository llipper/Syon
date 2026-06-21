"""CWE Detection benchmark — migrado de eval/benchmarks/cwe.py."""

from __future__ import annotations

from dataclasses import dataclass

from syon.security.analyzer import SecurityAnalyzer
from evaluation.metrics.security_metrics import cwe_classification_accuracy


@dataclass
class BenchmarkResult:
    name: str
    score: float
    note: str
    passed: int
    total: int


class CWEDetectionBenchmark:
    """Detecção de CWE em amostras de código."""

    TARGET_SCORE = 91.2
    NOTE = "Excelente"

    def __init__(self):
        self.analyzer = SecurityAnalyzer()

    def evaluate(self, samples: list[dict] | None = None) -> BenchmarkResult:
        samples = samples or self._default_samples()
        passed = 0
        predicted: list[str] = []
        expected: list[str] = []

        for sample in samples:
            result = self.analyzer.analyze(sample["code"], sample["language"])
            cwes = {f.cwe for f in result.findings}
            detected = sample["expected_cwe"] in cwes
            if detected:
                passed += 1
            predicted.append(next(iter(cwes), "none") if cwes else "none")
            expected.append(sample["expected_cwe"])

        total = len(samples)
        score = cwe_classification_accuracy(predicted, expected) * 100
        return BenchmarkResult(
            name="CWE Identification",
            score=round(score, 1),
            note=self.NOTE,
            passed=passed,
            total=total,
        )

    def _default_samples(self) -> list[dict]:
        return [
            {
                "code": "import pickle; pickle.loads(data)",
                "language": "python",
                "expected_cwe": "CWE-502",
            },
            {
                "code": "os.system('ls ' + filename)",
                "language": "python",
                "expected_cwe": "CWE-78",
            },
            {
                "code": "query = 'SELECT * FROM users WHERE id=' + user_id",
                "language": "python",
                "expected_cwe": "CWE-89",
            },
        ]


# Backward compatibility alias
CWEBenchmark = CWEDetectionBenchmark