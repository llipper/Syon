"""Métricas de linguagem natural."""

from __future__ import annotations

import math
from collections import Counter


def perplexity(loss: float) -> float:
    """Calcula perplexidade a partir da loss."""
    return math.exp(min(loss, 20.0))


def bleu_score(reference: str, hypothesis: str, n: int = 4) -> float:
    """BLEU score simplificado (n-gram precision)."""
    ref_tokens = reference.split()
    hyp_tokens = hypothesis.split()
    if not hyp_tokens:
        return 0.0

    precisions = []
    for i in range(1, n + 1):
        ref_ngrams = Counter(tuple(ref_tokens[j : j + i]) for j in range(len(ref_tokens) - i + 1))
        hyp_ngrams = Counter(tuple(hyp_tokens[j : j + i]) for j in range(len(hyp_tokens) - i + 1))
        if not hyp_ngrams:
            precisions.append(0.0)
            continue
        matches = sum(min(count, ref_ngrams[ng]) for ng, count in hyp_ngrams.items())
        precisions.append(matches / sum(hyp_ngrams.values()))

    if not precisions or 0 in precisions:
        return 0.0
    geo_mean = math.exp(sum(math.log(p) for p in precisions) / len(precisions))
    bp = min(1.0, math.exp(1 - len(ref_tokens) / max(len(hyp_tokens), 1)))
    return geo_mean * bp


def rouge_score(reference: str, hypothesis: str) -> float:
    """ROUGE-L F1 simplificado baseado em LCS."""
    ref_tokens = reference.split()
    hyp_tokens = hypothesis.split()
    if not ref_tokens or not hyp_tokens:
        return 0.0

    m, n = len(ref_tokens), len(hyp_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_tokens[i - 1] == hyp_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    lcs = dp[m][n]
    precision = lcs / n
    recall = lcs / m
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def exact_match(predictions: list[str], references: list[str]) -> float:
    """Exact match score."""
    if not references:
        return 0.0
    matches = sum(1 for p, r in zip(predictions, references) if p.strip() == r.strip())
    return matches / len(references)


def f1_score(predictions: list[str], references: list[str]) -> float:
    """Token-level F1 score."""
    if not references:
        return 0.0
    scores = []
    for pred, ref in zip(predictions, references):
        pred_tokens = set(pred.split())
        ref_tokens = set(ref.split())
        if not pred_tokens and not ref_tokens:
            scores.append(1.0)
            continue
        if not pred_tokens or not ref_tokens:
            scores.append(0.0)
            continue
        common = pred_tokens & ref_tokens
        precision = len(common) / len(pred_tokens)
        recall = len(common) / len(ref_tokens)
        if precision + recall == 0:
            scores.append(0.0)
        else:
            scores.append(2 * precision * recall / (precision + recall))
    return sum(scores) / len(scores)