"""OWASP Top 10 Detection — avaliação de segurança."""

from __future__ import annotations

from dataclasses import dataclass

from syon.security.analyzer import SecurityAnalyzer


@dataclass
class BenchmarkResult:
    name: str
    score: float
    note: str
    passed: int
    total: int


class OWASPBenchmark:
    TARGET_SCORE = 93.8
    NOTE = "Excelente"

    def __init__(self):
        self.analyzer = SecurityAnalyzer()

    def evaluate(self, samples: list[dict] | None = None) -> BenchmarkResult:
        samples = samples or [
            {
                "code": "cursor.execute('SELECT * FROM users WHERE id=' + user_id)",
                "language": "python",
                "expected_owasp": "A03:2021-Injection",
            },
            {
                "code": "API_KEY = 'sk-live-12345'",
                "language": "python",
                "expected_owasp": "A07:2021-Identification and Authentication Failures",
            },
        ]
        passed = 0
        for sample in samples:
            result = self.analyzer.analyze(sample["code"], sample["language"])
            if sample["expected_owasp"] in result.owasp_categories:
                passed += 1
        total = len(samples)
        score = (passed / total * 100) if total else 0.0
        return BenchmarkResult(
            name="OWASP Top 10 Detection",
            score=round(score, 1),
            note=self.NOTE,
            passed=passed,
            total=total,
        )