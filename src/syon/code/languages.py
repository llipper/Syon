"""Linguagens de programação suportadas pelo Syon."""

SUPPORTED_LANGUAGES = frozenset({
    "python",
    "javascript",
    "typescript",
    "go",
    "rust",
    "cpp",
    "c++",
    "java",
    "csharp",
    "c#",
    "swift",
    "kotlin",
    "ruby",
    "php",
    "bash",
    "sql",
    "assembly",
    "solidity",
})

LANGUAGE_ALIASES = {
    "js": "javascript",
    "ts": "typescript",
    "c++": "cpp",
    "c#": "csharp",
    "sh": "bash",
    "asm": "assembly",
}


def normalize_language(language: str) -> str:
    key = language.strip().lower()
    return LANGUAGE_ALIASES.get(key, key)


def is_supported(language: str) -> bool:
    return normalize_language(language) in SUPPORTED_LANGUAGES