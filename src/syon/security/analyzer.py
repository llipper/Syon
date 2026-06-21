"""Compatibilidade: re-exporta SAST do módulo security/."""

from security.vulnerability_scanning.sast import (
    SecurityAnalysisResult,
    SecurityAnalyzer,
    VulnerabilityFinding,
)

__all__ = ["SecurityAnalyzer", "SecurityAnalysisResult", "VulnerabilityFinding"]