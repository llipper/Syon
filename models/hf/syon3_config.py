"""Config Hugging Face para arquitetura Syon 3."""

from __future__ import annotations

from transformers import PretrainedConfig

from models.architecture.config import SyonModelConfig


class Syon3HFConfig(PretrainedConfig):
    model_type = "syon_3"

    def __init__(
        self,
        vocab_size: int = 8192,
        hidden_size: int = 384,
        num_layers: int = 6,
        num_heads: int = 6,
        intermediate_size: int = 1536,
        max_seq_length: int = 512,
        rope_theta: float = 10000.0,
        dropout: float = 0.1,
        tie_word_embeddings: bool = True,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        **kwargs,
    ) -> None:
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.intermediate_size = intermediate_size
        self.max_seq_length = max_seq_length
        self.rope_theta = rope_theta
        self.dropout = dropout
        self.tie_word_embeddings = tie_word_embeddings
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

    def to_syon_config(self) -> SyonModelConfig:
        return SyonModelConfig.from_dict(
            {
                "vocab_size": self.vocab_size,
                "hidden_size": self.hidden_size,
                "num_layers": self.num_layers,
                "num_heads": self.num_heads,
                "intermediate_size": self.intermediate_size,
                "max_seq_length": self.max_seq_length,
                "rope_theta": self.rope_theta,
                "dropout": self.dropout,
                "tie_word_embeddings": self.tie_word_embeddings,
                "pad_token_id": self.pad_token_id,
                "bos_token_id": self.bos_token_id,
                "eos_token_id": self.eos_token_id,
                "model_type": "syon_3",
            }
        )

    @classmethod
    def from_syon_config(cls, cfg: SyonModelConfig) -> Syon3HFConfig:
        return cls(**cfg.to_dict())