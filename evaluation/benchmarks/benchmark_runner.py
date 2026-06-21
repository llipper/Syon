"""Benchmark runner — migrado de eval/runner.py."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from typing import Any, Callable

from evaluation.benchmarks.code_benchmarks.humaneval import HumanEvalBenchmark
from evaluation.benchmarks.code_benchmarks.mbpp import MBPPBenchmark
from evaluation.benchmarks.security_benchmarks.cwe_detection import CWEDetectionBenchmark
from evaluation.benchmarks.security_benchmarks.vulnerability_analysis import VulnerabilityAnalysisBenchmark


def run_benchmark_suite(
    generate_fn: Callable[[str], str] | None = None,
    mbpp_fn: Callable[[str, str], str | bool] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Executa suite de benchmarks programação e segurança."""
    generate_fn = generate_fn or (lambda p: "def is_palindrome(s): return s == s[::-1]")
    mbpp_fn = mbpp_fn or (lambda p, l: True)

    results: dict[str, list[dict[str, Any]]] = {
        "programming": [],
        "security": [],
    }

    humaneval = HumanEvalBenchmark().evaluate(generate_fn)
    mbpp = MBPPBenchmark().evaluate(mbpp_fn)
    cwe = CWEDetectionBenchmark().evaluate()
    vuln = VulnerabilityAnalysisBenchmark().evaluate()

    results["programming"] = [asdict(humaneval), asdict(mbpp)]
    results["security"] = [asdict(cwe), asdict(vuln)]
    return results


def run_all_benchmarks() -> dict[str, list[dict[str, Any]]]:
    """Executa todos os benchmarks com funções padrão."""
    return run_benchmark_suite()


def compare_results(
    current: dict[str, list[dict]],
    baseline: dict[str, list[dict]],
) -> dict[str, list[dict[str, Any]]]:
    """Compara resultados atuais com baseline."""
    comparison: dict[str, list[dict[str, Any]]] = {}
    for category in current:
        comparison[category] = []
        baseline_scores = {r.get("name"): r.get("score", 0) for r in baseline.get(category, [])}
        for result in current.get(category, []):
            name = result.get("name", "unknown")
            current_score = result.get("score", 0)
            base_score = baseline_scores.get(name, 0)
            comparison[category].append({
                "name": name,
                "current_score": current_score,
                "baseline_score": base_score,
                "delta": round(current_score - base_score, 1),
                "improved": current_score > base_score,
            })
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(description="Syon Benchmark Runner")
    parser.add_argument("--output", type=str, default=None, help="Salvar resultados em JSON")
    parser.add_argument("--baseline", type=str, default=None, help="JSON de baseline para comparação")
    args = parser.parse_args()

    results = run_all_benchmarks()
    print(json.dumps(results, indent=2, ensure_ascii=False))

    if args.baseline:
        with open(args.baseline, encoding="utf-8") as handle:
            baseline = json.load(handle)
        comparison = compare_results(results, baseline)
        print("\nComparison:")
        print(json.dumps(comparison, indent=2, ensure_ascii=False))

    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()