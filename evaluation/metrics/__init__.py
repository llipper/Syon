"""Métricas de avaliação."""

from evaluation.metrics.code_metrics import code_duplication, correctness_score, cyclomatic_complexity, pass_at_k, test_coverage
from evaluation.metrics.language_metrics import bleu_score, exact_match, f1_score, perplexity, rouge_score
from evaluation.metrics.security_metrics import (
    attack_scenario_handling,
    cwe_classification_accuracy,
    cvss_score_correlation,
    false_positive_rate,
    vulnerability_detection_rate,
)

__all__ = [
    "pass_at_k",
    "correctness_score",
    "cyclomatic_complexity",
    "code_duplication",
    "test_coverage",
    "vulnerability_detection_rate",
    "false_positive_rate",
    "cwe_classification_accuracy",
    "cvss_score_correlation",
    "attack_scenario_handling",
    "perplexity",
    "bleu_score",
    "rouge_score",
    "exact_match",
    "f1_score",
]