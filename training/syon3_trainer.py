"""
Syon 3 Trainer — treino do zero em fases (programação, cybersecurity, arquitetura, conversa).

Uso:
    python -m training.syon3_trainer --config training/configs/syon3.yaml
    python -m training.syon3_trainer --phase 2 --resume training/checkpoints/phase1/best
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from training.dataset.curator import CuratedSample, DatasetCurator
from training.dataset.composition import load_composition
from training.losses.security_aware import SecurityAwareLoss


@dataclass
class PhaseConfig:
    name: str
    max_steps: int
    learning_rate: float
    security_aware_weight: float
    focus_domains: list[str]
    checkpoint_subdir: str


class Syon3Dataset(torch.utils.data.Dataset):
    def __init__(self, samples: list[CuratedSample], tokenizer, max_length: int, focus_domains: list[str] | None):
        if focus_domains:
            filtered = [s for s in samples if s.domain in focus_domains]
            self.samples = filtered if filtered else samples
        else:
            self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        enc = self.tokenizer(
            self.samples[idx].text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        ids = enc["input_ids"].squeeze(0)
        mask = enc["attention_mask"].squeeze(0)
        labels = ids.clone()
        labels[mask == 0] = -100
        return {"input_ids": ids, "attention_mask": mask, "labels": labels}


def collate(batch: list[dict]) -> dict:
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}


def _save_checkpoint(model, tokenizer, path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    if hasattr(model, "save_pretrained"):
        model.save_pretrained(path)
    if hasattr(tokenizer, "save_pretrained"):
        tokenizer.save_pretrained(path)


def load_syon3_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def build_scratch_model(config: dict, data_dir: Path | None = None, resume: Path | None = None):
    """Syon 3 + BPE — arquitetura e tokenizer proprietários, pesos aleatórios."""
    from models.architecture.config import PRESETS, SyonModelConfig
    from models.architecture.syon3 import Syon3
    from models.tokenizer.syon_bpe import SyonBPETokenizer

    mc = config.get("model", {})
    preset = mc.get("architecture", "syon3")
    max_len = int(mc.get("max_seq_length", 512))
    tok_dir = Path(mc.get("tokenizer_dir", ROOT / "models/tokenizer/syon-bpe"))

    if resume and (resume / "tokenizer.json").exists():
        tokenizer = SyonBPETokenizer.from_pretrained(resume)
    elif tok_dir.exists() and (tok_dir / "tokenizer.json").exists():
        tokenizer = SyonBPETokenizer.from_pretrained(tok_dir)
    else:
        if data_dir is None:
            data_dir = Path(config.get("data", {}).get("raw_dir", ROOT / "data/raw"))
        tokenizer = SyonBPETokenizer.train_from_corpus_dir(
            data_dir,
            vocab_size=int(mc.get("vocab_size", 8192)),
            max_length=max_len,
        )
        tok_dir.mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(tok_dir)

    base_cfg = PRESETS.get(preset, SyonModelConfig())
    cfg_dict = {**base_cfg.to_dict(), "vocab_size": tokenizer.vocab_size, "max_seq_length": max_len}
    cfg = SyonModelConfig.from_dict(cfg_dict)

    if resume and (resume / "config.json").exists():
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = Syon3.from_pretrained(resume, device=device)
    else:
        model = Syon3(cfg)

    print(
        f"[Syon 3] {preset} | params={model.num_parameters():,} | "
        f"vocab={tokenizer.vocab_size} | from_scratch=True"
    )
    return tokenizer, model, max_len


def build_model(config: dict, data_dir: Path | None = None, resume: Path | None = None):
    """Syon 3 exclusivamente — arquitetura e pesos proprietários."""
    return build_scratch_model(config, data_dir=data_dir, resume=resume)


def run_phase(
    model,
    tokenizer,
    samples: list[CuratedSample],
    phase: PhaseConfig,
    config: dict,
    device: torch.device,
    start_step: int = 0,
) -> int:
    tcfg = config.get("training", {})
    batch_size = int(tcfg.get("batch_size", 2))
    accum = int(tcfg.get("gradient_accumulation_steps", 8))
    save_every = int(tcfg.get("save_every_steps", 500))

    loader = DataLoader(
        Syon3Dataset(samples, tokenizer, int(config["model"]["max_seq_length"]), phase.focus_domains),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate,
        drop_last=False,
    )
    batches = max(1, len(loader))
    if accum > batches:
        accum = batches

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=phase.learning_rate,
        weight_decay=float(tcfg.get("weight_decay", 0.01)),
    )
    sec_loss = SecurityAwareLoss(weight=phase.security_aware_weight)
    use_cuda = device.type == "cuda"
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)
        autocast = lambda: torch.amp.autocast("cuda", enabled=use_cuda)
    except AttributeError:
        scaler = torch.cuda.amp.GradScaler(enabled=use_cuda)
        autocast = lambda: torch.cuda.amp.autocast(enabled=use_cuda)

    ckpt_base = Path(tcfg.get("checkpoint_dir", "training/checkpoints")) / phase.checkpoint_subdir
    ckpt_base.mkdir(parents=True, exist_ok=True)

    model.train()
    step = start_step
    micro = 0
    running = 0.0
    optimizer.zero_grad()
    max_epochs = int(tcfg.get("max_epochs", 100))

    print(f"\n{'='*60}")
    print(f"FASE: {phase.name} | steps alvo: {phase.max_steps} | domains: {phase.focus_domains}")
    print(f"{'='*60}")

    epoch = 0
    while step < phase.max_steps and epoch < max_epochs:
        epoch += 1
        for batch in loader:
            if step >= phase.max_steps:
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            with autocast():
                out = model(**batch)
                ce = F.cross_entropy(
                    out.logits.view(-1, out.logits.size(-1)),
                    batch["labels"].view(-1),
                    ignore_index=-100,
                )
                sec = sec_loss(out.logits, batch["labels"])
                loss = (ce + sec) / accum

            scaler.scale(loss).backward()
            running += float(loss.item()) * accum
            micro += 1

            if micro % accum == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                step += 1
                if step % 10 == 0 or step == 1:
                    print(f"  [{phase.name}] step {step}/{phase.max_steps} loss={running/accum:.4f}")
                running = 0.0

                if step % save_every == 0:
                    p = ckpt_base / f"step_{step}"
                    _save_checkpoint(model, tokenizer, p)

        if micro % accum != 0 and step < phase.max_steps:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            step += 1
            micro = 0

    best = ckpt_base / "best"
    best.mkdir(parents=True, exist_ok=True)
    _save_checkpoint(model, tokenizer, best)
    print(f"  [{phase.name}] checkpoint final: {best}")
    return step


def ensure_curriculum(data_dir: Path, config: dict) -> None:
    manifest = data_dir.parent / "curriculum" / "build_manifest.json"
    data_cfg = config.get("data", {})
    min_samples = int(data_cfg.get("min_samples", 10000))
    augment = int(data_cfg.get("curriculum_augment", 3))

    if manifest.exists():
        try:
            info = json.loads(manifest.read_text(encoding="utf-8"))
            if int(info.get("total_samples", 0)) >= min_samples:
                print(f"[Syon 3] Dataset OK: {info['total_samples']} amostras")
                return
        except (json.JSONDecodeError, KeyError):
            pass

    print(f"[Syon 3] Gerando curriculum (mín. {min_samples})...")
    from scripts.data.build_syon3_curriculum import build

    build(data_dir, min_samples=min_samples, augment=augment)


def main() -> None:
    parser = argparse.ArgumentParser(description="Syon 3 Trainer")
    parser.add_argument("--config", type=Path, default=ROOT / "training/configs/syon3.yaml")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/raw")
    parser.add_argument("--phase", type=int, default=0, help="0=todas, 1/2/3/4=fase específica")
    parser.add_argument("--augment", type=int, default=20, help="Fator curriculum se dados ausentes")
    parser.add_argument("--resume", type=Path, default=None)
    args = parser.parse_args()

    config = load_syon3_config(args.config)
    data_dir = Path(args.data_dir or config.get("data", {}).get("raw_dir", ROOT / "data/raw"))
    ensure_curriculum(data_dir, config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("[Syon 3] AVISO: sem GPU — treino requer CUDA")

    print(f"[Syon 3] Device: {device}")
    resume_path = args.resume if args.resume and args.resume.exists() else None
    tokenizer, model, _ = build_model(config, data_dir=data_dir, resume=resume_path)
    model = model.to(device)
    if resume_path:
        print(f"[Syon 3] Resumido de {resume_path}")

    curator = DatasetCurator(data_dir, load_composition())
    samples = curator.curate_all()
    print(f"[Syon 3] Dataset: {len(samples)} amostras")
    comp = load_composition()
    print(f"[Syon 3] Composição OK: {comp.validate_weights()}")

    phases_cfg = config.get("phases", [])
    phases = [
        PhaseConfig(
            name=p["name"],
            max_steps=int(p["max_steps"]),
            learning_rate=float(p["learning_rate"]),
            security_aware_weight=float(p.get("security_aware_weight", 0.15)),
            focus_domains=list(p.get("focus_domains", [])),
            checkpoint_subdir=p.get("checkpoint_subdir", p["name"]),
        )
        for p in phases_cfg
    ]

    run_phases = phases if args.phase == 0 else [phases[args.phase - 1]]

    results: dict[str, Any] = {"phases": []}
    for phase in run_phases:
        steps = run_phase(model, tokenizer, samples, phase, config, device)
        results["phases"].append({"name": phase.name, "steps": steps})

    out_model = Path(config.get("output", {}).get("pretrained_dir", "models/pretrained/syon-3"))
    out_model.mkdir(parents=True, exist_ok=True)
    _save_checkpoint(model, tokenizer, out_model)

    results["output"] = str(out_model)
    results["total_samples"] = len(samples)
    log_path = Path(config.get("output", {}).get("summary", "training/logs/syon3_summary.json"))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"\n[Syon 3] Treino completo. Modelo: {out_model}")
    print(f"[Syon 3] Resumo: {log_path}")


if __name__ == "__main__":
    main()