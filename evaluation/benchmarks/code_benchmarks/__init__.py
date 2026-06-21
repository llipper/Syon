"""Benchmarks de código."""

from evaluation.benchmarks.code_benchmarks.humaneval import HumanEvalBenchmark, load_humaneval_dataset
from evaluation.benchmarks.code_benchmarks.mbpp import MBPPBenchmark

__all__ = ["HumanEvalBenchmark", "load_humaneval_dataset", "MBPPBenchmark"]