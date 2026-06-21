"""
Política Hugging Face — APENAS infraestrutura de treino.

Permitido:
  - datasets.load_dataset  → baixar dados de TEXTO (diálogos PT)
  - transformers.Trainer   → loop de treino do Syon 3

Proibido:
  - Carregar pesos de modelos de terceiros (Gemma, Llama, GPT, Claude, etc.)
  - AutoModel.from_pretrained de repositórios externos
  - LoRA / distillation sobre modelos externos
"""

from __future__ import annotations

# Repositórios de DADOS (texto) permitidos via API HF
ALLOWED_HF_DATASETS: frozenset[str] = frozenset({
    "nicholasKluge/instruct-aira-dataset-v3",
})

BLOCKED_MODEL_PREFIXES: tuple[str, ...] = (
    "google/gemma",
    "meta-llama",
    "mistralai",
    "Qwen",
    "microsoft/phi",
    "openai",
    "anthropic",
)


def assert_dataset_allowed(repo_id: str | None) -> None:
    if not repo_id:
        return
    if repo_id not in ALLOWED_HF_DATASETS:
        raise ValueError(
            f"Dataset HF não autorizado: {repo_id}\n"
            f"Permitidos: {sorted(ALLOWED_HF_DATASETS)}\n"
            "Use dados locais em data/raw/ ou adicione o repo à política."
        )


def assert_no_external_model(repo_id: str) -> None:
    lower = repo_id.lower()
    for prefix in BLOCKED_MODEL_PREFIXES:
        if lower.startswith(prefix.lower()):
            raise ValueError(
                f"Modelo externo bloqueado: {repo_id}\n"
                "Syon 3 é treinado do zero — sem pesos de outras IAs."
            )