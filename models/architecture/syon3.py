"""
Syon 3 — LLM decoder-only proprietário (treinado do zero).

Arquitetura Syon: Pre-norm RMSNorm + RoPE + SwiGLU FFN + causal self-attention.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.architecture.config import PRESETS, SyonModelConfig


@dataclass
class Syon3Output:
    loss: torch.Tensor | None
    logits: torch.Tensor


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.pow(2).mean(dim=-1, keepdim=True)
        return self.weight * x * torch.rsqrt(norm + self.eps)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    q = (q * cos) + (rotate_half(q) * sin)
    k = (k * cos) + (rotate_half(k) * sin)
    return q, k


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq: int, theta: float) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq)

    def _build_cache(self, seq_len: int) -> None:
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, seq_len: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.cos_cached.shape[2]:
            self._build_cache(seq_len)
        return (
            self.cos_cached[:, :, :seq_len, :].to(device),
            self.sin_cached[:, :, :seq_len, :].to(device),
        )


class CausalSelfAttention(nn.Module):
    def __init__(self, config: SyonModelConfig) -> None:
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.hidden = config.hidden_size
        self.qkv = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=False)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        self.rope = RotaryEmbedding(self.head_dim, config.max_seq_length, config.rope_theta)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rope(seq_len, x.device)
        q, k = apply_rope(q, k, cos, sin)

        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        causal = torch.triu(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), diagonal=1)
        attn = attn.masked_fill(causal, float("-inf"))
        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask[:, None, None, :] == 0, float("-inf"))

        weights = F.softmax(attn, dim=-1)
        weights = self.dropout(weights)
        out = weights @ v
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, self.hidden)
        return self.out_proj(out)


class SwiGLU(nn.Module):
    def __init__(self, config: SyonModelConfig) -> None:
        super().__init__()
        self.gate = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down(F.silu(self.gate(x)) * self.up(x)))


class TransformerBlock(nn.Module):
    def __init__(self, config: SyonModelConfig) -> None:
        super().__init__()
        self.norm1 = RMSNorm(config.hidden_size)
        self.attn = CausalSelfAttention(config)
        self.norm2 = RMSNorm(config.hidden_size)
        self.mlp = SwiGLU(config)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), attn_mask)
        x = x + self.mlp(self.norm2(x))
        return x


class Syon3(nn.Module):
    """Syon 3 — modelo causal LM proprietário, pesos inicializados do zero."""

    def __init__(self, config: SyonModelConfig) -> None:
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])
        self.norm = RMSNorm(config.hidden_size)
        if config.tie_word_embeddings:
            self.lm_head: nn.Module | None = None
        else:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> Syon3Output:
        x = self.embed(input_ids)
        for block in self.blocks:
            x = block(x, attention_mask)
        x = self.norm(x)
        logits = x @ self.embed.weight.T if self.config.tie_word_embeddings else self.lm_head(x)  # type: ignore[operator]

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )
        return Syon3Output(loss=loss, logits=logits)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def save_pretrained(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self.config.save(path / "config.json")
        torch.save(self.state_dict(), path / "pytorch_model.bin")
        meta = {
            "model_name": "Syon 3",
            "model_type": "syon_3",
            "from_scratch": True,
            "parameters": self.num_parameters(),
        }
        (path / "syon_model.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    @classmethod
    def from_pretrained(cls, path: Path, device: str | torch.device = "cpu") -> Syon3:
        config = SyonModelConfig.load(path / "config.json")
        model = cls(config)
        weights = path / "pytorch_model.bin"
        if not weights.exists():
            raise FileNotFoundError(f"Pesos não encontrados: {weights}")
        state = torch.load(weights, map_location=device, weights_only=True)
        model.load_state_dict(state)
        return model.to(device)

    @classmethod
    def from_preset(cls, name: str) -> Syon3:
        if name not in PRESETS:
            raise KeyError(f"Preset desconhecido: {name}. Opções: {list(PRESETS)}")
        return cls(PRESETS[name])

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed