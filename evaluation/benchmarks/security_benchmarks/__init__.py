"""Benchmarks de segurança."""

from evaluation.benchmarks.security_benchmarks.cwe_detection import CWEBenchmark, CWEDetectionBenchmark
from evaluation.benchmarks.security_benchmarks.vulnerability_analysis import VulnerabilityAnalysisBenchmark

__all__ = ["CWEBenchmark", "CWEDetectionBenchmark", "VulnerabilityAnalysisBenchmark"]