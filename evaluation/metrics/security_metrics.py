"""Métricas de segurança."""

from __future__ import annotations

from typing import Any


def vulnerability_detection_rate(
    predictions: list[bool],
    ground_truth: list[bool],
) -> float:
    """Taxa de detecção de vulnerabilidades (recall)."""
    if not ground_truth:
        return 0.0
    true_positives = sum(1 for p, g in zip(predictions, ground_truth) if p and g)
    actual_positives = sum(ground_truth)
    return true_positives / actual_positives if actual_positives else 0.0


def false_positive_rate(
    predictions: list[bool],
    ground_truth: list[bool],
) -> float:
    """Taxa de falsos positivos."""
    if not predictions:
        return 0.0
    false_positives = sum(1 for p, g in zip(predictions, ground_truth) if p and not g)
    actual_negatives = sum(1 for g in ground_truth if not g)
    return false_positives / actual_negatives if actual_negatives else 0.0


def cwe_classification_accuracy(
    predicted_cwes: list[str],
    expected_cwes: list[str],
) -> float:
    """Acurácia de classificação CWE."""
    if not expected_cwes:
        return 0.0
    correct = sum(1 for p, e in zip(predicted_cwes, expected_cwes) if p == e)
    return correct / len(expected_cwes)


def cvss_score_correlation(
    predicted_scores: list[float],
    actual_scores: list[float],
) -> float:
    """Correlação entre scores CVSS previstos e reais."""
    if len(predicted_scores) < 2:
        return 0.0
    n = len(predicted_scores)
    mean_p = sum(predicted_scores) / n
    mean_a = sum(actual_scores) / n
    num = sum((p - mean_p) * (a - mean_a) for p, a in zip(predicted_scores, actual_scores))
    den_p = sum((p - mean_p) ** 2 for p in predicted_scores) ** 0.5
    den_a = sum((a - mean_a) ** 2 for a in actual_scores) ** 0.5
    if den_p * den_a == 0:
        return 0.0
    return num / (den_p * den_a)


def attack_scenario_handling(
    responses: list[str],
    scenarios: list[dict[str, Any]],
) -> float:
    """Avalia se respostas tratam cenários de ataque adequadamente."""
    if not scenarios:
        return 0.0
    handled = 0
    for response, scenario in zip(responses, scenarios):
        keywords = scenario.get("expected_keywords", [])
        if any(kw.lower() in response.lower() for kw in keywords):
            handled += 1
    return handled / len(scenarios)