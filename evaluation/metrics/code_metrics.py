"""Métricas de código."""

from __future__ import annotations

import math
import random
import re
from typing import Callable


def pass_at_k(
    n: int,
    c: int,
    k: int,
) -> float:
    """
    Calcula Pass@k (HumanEval).
    n: total de amostras geradas
    c: número de amostras corretas
    k: k em pass@k
    """
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def correctness_score(generated: str, expected: str) -> float:
    """Score de correção normalizado (0-1)."""
    return 1.0 if generated.strip() == expected.strip() else 0.0


def cyclomatic_complexity(code: str) -> int:
    """Estimativa simplificada de complexidade ciclomática."""
    keywords = ["if", "elif", "for", "while", "except", "and", "or", "case", "when"]
    count = 1
    for kw in keywords:
        count += len(re.findall(rf"\b{kw}\b", code))
    return count


def code_duplication(code_blocks: list[str]) -> float:
    """Taxa de duplicação entre blocos de código (0-1)."""
    if len(code_blocks) < 2:
        return 0.0
    duplicates = 0
    seen: set[str] = set()
    for block in code_blocks:
        normalized = re.sub(r"\s+", " ", block.strip())
        if normalized in seen:
            duplicates += 1
        seen.add(normalized)
    return duplicates / len(code_blocks)


def test_coverage(solutions: list[str], tests: list[str]) -> float:
    """Taxa de soluções que passam nos testes."""
    if not solutions:
        return 0.0
    passed = 0
    for solution, test in zip(solutions, tests):
        try:
            namespace: dict = {}
            exec(solution, namespace)
            exec(test, namespace)
            passed += 1
        except Exception:
            continue
    return passed / len(solutions)


def evaluate_pass_at_k(
    problems: list[dict],
    generate_fn: Callable[[str], str],
    k: int = 1,
    n_samples: int = 1,
) -> float:
    """Avalia pass@k em lista de problemas."""
    scores = []
    for problem in problems:
        prompt = problem.get("prompt", "")
        test = problem.get("test", "")
        correct = 0
        for _ in range(n_samples):
            output = generate_fn(prompt)
            try:
                namespace: dict = {}
                exec(output, namespace)
                exec(test, namespace)
                correct += 1
            except Exception:
                pass
        scores.append(pass_at_k(n_samples, correct, k))
    return sum(scores) / len(scores) if scores else 0.0