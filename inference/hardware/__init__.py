"""Suporte a hardware para inferência."""

from inference.hardware.cpu import estimate_cpu_time, optimize_for_cpu
from inference.hardware.gpu import allocate_gpu_memory, cuda_optimizations, detect_gpu

__all__ = [
    "detect_gpu",
    "allocate_gpu_memory",
    "cuda_optimizations",
    "optimize_for_cpu",
    "estimate_cpu_time",
]