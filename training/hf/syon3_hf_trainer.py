"""
Syon 3 — treino via Hugging Face (APENAS infraestrutura).

Treina o Syon 3 proprietário do zero a:
  - Conversar em português
  - Raciocinar (programação, cybersecurity, arquitetura)

Hugging Face fornece: Trainer + API de datasets (texto).
NÃO carrega modelos de terceiros.

Uso:
    python -m training.hf.syon3_hf_trainer
    python -m training.hf.syon3_hf_trainer --resume kl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
import yaml
from transformers import Trainer, TrainingArguments

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "training/configs/syon3_hf.yaml"

for p in (ROOT, ROOT / "src"):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)

from models.hf.syon3_model import Syon3ForCausalLM
from models.tokenizer.syon_bpe import SyonBPETokenizer
from training.hf.policy import assert_no_external_model
from training.hf.training_data import load_syon3_hf_dataset

DEFAULT_TOKENIZER_CANDIDATES = ("kl", "models/tokenizer/syon3-bpe", "models/pretrained/syon-3")
DEFAULT_CHECKPOINT_CANDIDATES = (
    "kl",
    "models/pretrained/syon-3",
    "training/checkpoints/phase4/best",
)


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_path(value: str | Path | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _unique_paths(*groups: Path | None) -> list[Path]:
    seen: set[Path] = set()
    ordered: list[Path] = []
    for group in groups:
        if group is None:
            continue
        p = group.resolve()
        if p not in seen:
            seen.add(p)
            ordered.append(p)
    return ordered


def find_tokenizer_dir(
    config: dict[str, Any],
    *,
    resume: Path | None,
    tokenizer_dir: Path | None,
) -> Path:
    mc = config.get("model", {})
    candidates = _unique_paths(
        resume,
        tokenizer_dir,
        resolve_path(mc.get("tokenizer_dir")),
        resolve_path(mc.get("resume")),
        *(resolve_path(p) for p in DEFAULT_TOKENIZER_CANDIDATES),
    )
    for path in candidates:
        if (path / "tokenizer.json").exists():
            print(f"[Syon 3/HF] Tokenizer: {path}")
            return path
    raise FileNotFoundError(
        "Tokenizer Syon 3 não encontrado. Coloque tokenizer.json em kl/"
    )


def find_checkpoint_dir(config: dict[str, Any], *, resume: Path | None) -> Path | None:
    mc = config.get("model", {})
    candidates = _unique_paths(
        resume,
        resolve_path(mc.get("resume")),
        *(resolve_path(p) for p in DEFAULT_CHECKPOINT_CANDIDATES),
    )
    for path in candidates:
        if (path / "pytorch_model.bin").exists() and (path / "config.json").exists():
            print(f"[Syon 3/HF] Checkpoint: {path}")
            return path
    return None


def load_model(config: dict[str, Any], resume: Path | None) -> Syon3ForCausalLM:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = find_checkpoint_dir(config, resume=resume)
    if checkpoint:
        return Syon3ForCausalLM.from_syon_checkpoint(checkpoint, device=device)

    from models.architecture.config import PRESETS, SyonModelConfig
    from models.hf.syon3_config import Syon3HFConfig

    preset = config.get("model", {}).get("architecture", "syon3")
    syon_cfg = PRESETS.get(preset, SyonModelConfig())
    print(f"[Syon 3/HF] Pesos aleatórios — arquitetura proprietária ({preset})")
    return Syon3ForCausalLM(Syon3HFConfig.from_syon_config(syon_cfg))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Syon 3 — conversação + raciocínio (HF Trainer, modelo proprietário)"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--tokenizer-dir", type=Path, default=None)
    parser.add_argument("--max-samples", type=int, default=None, help="Limite total (teste)")
    parser.add_argument("--push-to-hub", type=str, default=None, help="Publicar Syon 3 (opcional)")
    args = parser.parse_args()

    if args.push_to_hub:
        assert_no_external_model(args.push_to_hub)

    config = load_config(args.config.resolve())
    mc = config.get("model", {})
    tc = config.get("training", {})
    dc = config.get("data", {})
    oc = config.get("output", {})

    resume = resolve_path(args.resume) if args.resume else None
    tokenizer_dir = resolve_path(args.tokenizer_dir) if args.tokenizer_dir else None

    tok_path = find_tokenizer_dir(config, resume=resume, tokenizer_dir=tokenizer_dir)
    tokenizer = SyonBPETokenizer.from_pretrained(tok_path)
    model = load_model(config, resume)

    conversation_dir = resolve_path(dc.get("conversation_dir", "data/raw/conversation"))
    raw_dir = resolve_path(dc.get("raw_dir", "data/raw"))
    assert conversation_dir and raw_dir

    max_conv = int(dc.get("max_conversation_samples", 100_000))
    max_reas = int(dc.get("max_reasoning_samples", 50_000))
    if args.max_samples:
        max_conv = args.max_samples // 2
        max_reas = args.max_samples // 2

    max_length = int(mc.get("max_seq_length", tokenizer.max_length))
    train_ds, eval_ds = load_syon3_hf_dataset(
        tokenizer,
        conversation_dir=conversation_dir,
        raw_dir=raw_dir,
        max_length=max_length,
        max_conversation=max_conv,
        max_reasoning=max_reas,
        reasoning_domains=list(dc.get("reasoning_domains", ["programming", "cybersecurity", "complementary"])),
        hf_dataset=dc.get("hf_dataset"),
        hf_split=str(dc.get("hf_split", "portuguese")),
        conversation_weight=float(dc.get("conversation_weight", 0.6)),
        val_ratio=float(dc.get("val_ratio", 0.02)),
    )

    output_dir = resolve_path(tc.get("output_dir", "training/checkpoints/syon3_hf"))
    assert output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    use_cuda = torch.cuda.is_available()
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=int(tc.get("batch_size", 4)),
        per_device_eval_batch_size=int(tc.get("batch_size", 4)),
        gradient_accumulation_steps=int(tc.get("gradient_accumulation_steps", 4)),
        learning_rate=float(tc.get("learning_rate", 5e-5)),
        max_steps=int(tc.get("max_steps", 3000)),
        warmup_steps=int(tc.get("warmup_steps", 100)),
        weight_decay=float(tc.get("weight_decay", 0.01)),
        logging_steps=int(tc.get("logging_steps", 50)),
        save_steps=int(tc.get("save_steps", 200)),
        eval_strategy="steps" if eval_ds is not None else "no",
        eval_steps=int(tc.get("eval_steps", 200)) if eval_ds else None,
        save_total_limit=int(tc.get("save_total_limit", 3)),
        fp16=bool(tc.get("fp16", True)) and use_cuda,
        bf16=bool(tc.get("bf16", False)) and use_cuda,
        report_to=tc.get("report_to", "none"),
        remove_unused_columns=False,
        dataloader_pin_memory=use_cuda,
        seed=int(tc.get("seed", 42)),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
    )

    print(
        f"[Syon 3/HF] Treino | {len(train_ds)} amostras | "
        f"conversa+raciocínio | device={'cuda' if use_cuda else 'cpu'}"
    )
    trainer.train()

    model_dir = resolve_path(oc.get("model_dir", "models/pretrained/syon-3"))
    assert model_dir
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)

    summary = {
        "model": "syon_3",
        "from_scratch": True,
        "external_models": False,
        "hf_role": "trainer_and_datasets_api_only",
        "train_samples": len(train_ds),
        "eval_samples": len(eval_ds) if eval_ds else 0,
        "output": str(model_dir),
        "max_steps": training_args.max_steps,
    }
    summary_path = resolve_path(oc.get("summary", "training/logs/syon3_hf.json"))
    assert summary_path
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    hub_repo = args.push_to_hub or oc.get("hub_repo")
    if hub_repo:
        assert_no_external_model(hub_repo)
        model.push_to_hub(hub_repo)

    print(f"[Syon 3/HF] Concluído → {model_dir}")


if __name__ == "__main__":
    main()