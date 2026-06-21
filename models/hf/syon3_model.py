"""Wrapper Hugging Face do Syon 3 — compatível com Trainer."""

from __future__ import annotations

from pathlib import Path

import torch
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutput

from models.architecture.config import SyonModelConfig
from models.architecture.syon3 import Syon3
from models.hf.syon3_config import Syon3HFConfig


class Syon3ForCausalLM(PreTrainedModel):
    config_class = Syon3HFConfig

    def __init__(self, config: Syon3HFConfig) -> None:
        super().__init__(config)
        self.model = Syon3(config.to_syon_config())
        self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        **kwargs,
    ) -> CausalLMOutput:
        out = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return CausalLMOutput(loss=out.loss, logits=out.logits)

    def get_input_embeddings(self) -> torch.nn.Embedding:
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value: torch.nn.Embedding) -> None:
        self.model.embed = value

    def save_pretrained(self, save_directory: str | Path, **kwargs) -> None:
        path = Path(save_directory)
        path.mkdir(parents=True, exist_ok=True)
        self.config.save_pretrained(path)
        self.model.save_pretrained(path)

    @classmethod
    def from_syon_checkpoint(
        cls,
        checkpoint_dir: str | Path,
        device: str | torch.device = "cpu",
    ) -> Syon3ForCausalLM:
        path = Path(checkpoint_dir)
        syon_cfg = SyonModelConfig.load(path / "config.json")
        hf_cfg = Syon3HFConfig.from_syon_config(syon_cfg)
        wrapper = cls(hf_cfg)
        base = Syon3.from_pretrained(path, device=device)
        wrapper.model = base
        return wrapper