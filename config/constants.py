"""Constantes globais do projeto Syon."""

from __future__ import annotations

from typing import Final

SYON_VERSION: Final[str] = "1.0.0-beta"

MODEL_SIZES: Final[dict[str, dict[str, int | str]]] = {
    "syon-7b": {
        "parameters_billions": 7,
        "hidden_size": 4096,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "vocab_size": 32000,
        "context_window": 8192,
    },
    "syon-13b": {
        "parameters_billions": 13,
        "hidden_size": 5120,
        "num_hidden_layers": 40,
        "num_attention_heads": 40,
        "vocab_size": 32000,
        "context_window": 8192,
    },
    "syon-70b": {
        "parameters_billions": 70,
        "hidden_size": 8192,
        "num_hidden_layers": 80,
        "num_attention_heads": 64,
        "vocab_size": 32000,
        "context_window": 8192,
    },
}

# 15 linguagens listadas no README:
# Python, JavaScript/TypeScript, Go, Rust, C++, Java, C#, Swift, Kotlin,
# Ruby, PHP, Bash, SQL, Assembly, Solidity.
SUPPORTED_LANGUAGES: Final[tuple[str, ...]] = (
    "python",
    "javascript",
    "go",
    "rust",
    "cpp",
    "java",
    "csharp",
    "swift",
    "kotlin",
    "ruby",
    "php",
    "bash",
    "sql",
    "assembly",
    "solidity",
)

BENCHMARK_NAMES: Final[tuple[str, ...]] = (
    "humaneval",
    "mbpp",
    "code2seq",
    "apps",
    "cwe_detection",
    "vulnerability_analysis",
    "compliance_check",
    "mmlu",
    "arc",
    "hellaswag",
    "architecture_design",
    "security_scenarios",
)

DEFAULT_TIMEOUTS: Final[dict[str, int]] = {
    "api_request_seconds": 120,
    "inference_seconds": 300,
    "training_checkpoint_seconds": 600,
    "health_check_seconds": 5,
    "websocket_ping_seconds": 30,
    "batch_job_seconds": 3600,
    "security_analysis_seconds": 180,
    "model_load_seconds": 900,
}