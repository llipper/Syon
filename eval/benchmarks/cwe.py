"""CWE Identification — backward compatibility wrapper."""

from evaluation.benchmarks.security_benchmarks.cwe_detection import (
    BenchmarkResult,
    CWEBenchmark,
    CWEDetectionBenchmark,
)

__all__ = ["BenchmarkResult", "CWEBenchmark", "CWEDetectionBenchmark"]