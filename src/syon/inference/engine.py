"""Compatibilidade: re-exporta motor de inferência do módulo inference/."""

from inference.core.inference_engine import GenerationParams, InferenceEngine

__all__ = ["GenerationParams", "InferenceEngine"]