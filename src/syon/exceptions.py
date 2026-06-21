"""Exceções do pacote Syon."""


class SyonError(Exception):
    """Erro base do Syon."""


class ModelNotFoundError(SyonError):
    """Modelo ou arquivo de pesos não encontrado."""


class InferenceError(SyonError):
    """Falha durante inferência."""


class SecurityPolicyError(SyonError):
    """Requisição bloqueada por política de segurança."""


class UnsupportedLanguageError(SyonError):
    """Linguagem de programação não suportada."""