"""Orquestrador de benchmarks Syon — delega para evaluation/."""

from __future__ import annotations

from evaluation.benchmarks.benchmark_runner import compare_results, main, run_all_benchmarks, run_benchmark_suite

__all__ = ["run_all_benchmarks", "run_benchmark_suite", "compare_results", "main"]

if __name__ == "__main__":
    main()