"""
Pré-treino Syon do ZERO — causal language modeling com pesos aleatórios.

Uso:
    python -m training.pretrain --config training/configs/syon_scratch_kaggle.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "src"):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)

from training.master_trainer import MasterDataset, build_model, collate, load_master_config
from training.dataset.curator import DatasetCurator
from training.dataset.composition import load_composition


def run_pretrain(
    model,
    tokenizer,
    samples,
    config: dict[str, Any],
    device: torch.device,
) -> Path:
    pcfg = config.get("pretrain", {})
    max_steps = int(pcfg.get("max_steps", 1000))
    lr = float(pcfg.get("learning_rate", 3e-4))
    batch_size = int(pcfg.get("batch_size", config.get("training", {}).get("batch_size", 4)))
    accum = int(pcfg.get("gradient_accumulation_steps", config.get("training", {}).get("gradient_accumulation_steps", 4)))
    save_every = int(pcfg.get("save_every_steps", 100))
    max_len = int(config["model"]["max_seq_length"])

    loader = DataLoader(
        MasterDataset(samples, tokenizer, max_len, None),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate,
        drop_last=False,
    )
    batches = max(1, len(loader))
    if accum > batches:
        accum = batches

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    use_cuda = device.type == "cuda"
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)
        autocast = lambda: torch.amp.autocast("cuda", enabled=use_cuda)
    except AttributeError:
        scaler = torch.cuda.amp.GradScaler(enabled=use_cuda)
        autocast = lambda: torch.cuda.amp.autocast(enabled=use_cuda)

    ckpt_dir = Path(pcfg.get("checkpoint_dir", "training/checkpoints/pretrain"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    model.train()
    step = 0
    micro = 0
    running = 0.0
    optimizer.zero_grad()

    print(f"\n{'='*60}")
    print(f"[Syon/Pretrain] DO ZERO — causal LM | steps={max_steps} | params={model.num_parameters():,}")
    print(f"{'='*60}")

    epoch = 0
    while step < max_steps:
        epoch += 1
        for batch in loader:
            if step >= max_steps:
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            with autocast():
                out = model(**batch)
                loss = out.loss / accum if out.loss is not None else F.cross_entropy(
                    out.logits.view(-1, out.logits.size(-1)),
                    batch["labels"].view(-1),
                    ignore_index=-100,
                ) / accum

            scaler.scale(loss).backward()
            running += float(loss.item()) * accum
            micro += 1

            if micro % accum == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                step += 1
                if step % 10 == 0 or step == 1:
                    print(f"  [pretrain] step {step}/{max_steps} loss={running/accum:.4f}")
                running = 0.0
                if step % save_every == 0:
                    p = ckpt_dir / f"step_{step}"
                    model.save_pretrained(p)
                    tokenizer.save_pretrained(p)

    best = ckpt_dir / "best"
    best.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(best)
    tokenizer.save_pretrained(best)
    print(f"[Syon/Pretrain] ✓ Checkpoint: {best}")
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description="Syon pretrain do zero")
    parser.add_argument("--config", type=Path, default=ROOT / "training/configs/syon_scratch_kaggle.yaml")
    parser.add_argument("--data-dir", type=Path, default=None)
    args = parser.parse_args()

    config = load_master_config(args.config)
    data_dir = args.data_dir or Path(config.get("data", {}).get("raw_dir", ROOT / "data/raw"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Syon/Pretrain] Device: {device}")

    tokenizer, model, _ = build_model(config, data_dir=data_dir)
    model = model.to(device)

    curator = DatasetCurator(data_dir, load_composition())
    samples = curator.curate_all()
    print(f"[Syon/Pretrain] Corpus: {len(samples)} amostras")

    best = run_pretrain(model, tokenizer, samples, config, device)

    out = Path(config.get("output", {}).get("pretrain_dir", "models/pretrained/syon-pretrain"))
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    tokenizer.save_pretrained(out)

    summary = {"pretrain_steps": config.get("pretrain", {}).get("max_steps"), "checkpoint": str(best), "samples": len(samples)}
    log = Path(config.get("output", {}).get("summary", "training/logs/pretrain_summary.json"))
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[Syon/Pretrain] Modelo base: {out}")


if __name__ == "__main__":
    main()