"""Compatibilidade: re-exporta guardrails do módulo security/."""

from security.input_validation.injection_detection import check_input, should_refuse_topic
from security.output_filtering.content_filter import (
    MALICIOUS_OUTPUT_PATTERNS,
    check_output,
)

__all__ = [
    "MALICIOUS_OUTPUT_PATTERNS",
    "check_input",
    "check_output",
    "should_refuse_topic",
]