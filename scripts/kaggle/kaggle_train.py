"""
Treinamento Syon otimizado para Kaggle (GPU T4/P100 16GB).

- 1 modelo base + LoRA (QLoRA 4-bit) — cabe em 16GB
- Security-Aware Loss do pipeline Syon
- Dataset demo automático ou /kaggle/input/

Uso:
    python scripts/kaggle/kaggle_train.py --config training/configs/kaggle_config.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for p in (str(PROJECT_ROOT), str(PROJECT_ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)


def is_kaggle() -> bool:
    return Path("/kaggle").exists() or os.getenv("KAGGLE_KERNEL_RUN_TYPE") is not None


def detect_device() -> torch.device:
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"[Syon/Kaggle] GPU: {props.name} ({props.total_memory / 1e9:.1f} GB)")
        return torch.device("cuda")
    print("[Syon/Kaggle] AVISO: GPU não detectada!")
    return torch.device("cpu")


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def build_model(config: dict):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    model_cfg = config.get("model_params", {})
    base_model = model_cfg.get("base_model", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    max_len = int(model_cfg.get("max_seq_length", 512))
    use_lora = model_cfg.get("use_lora", True)

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = None
    if torch.cuda.is_available():
        try:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            print("[Syon/Kaggle] QLoRA 4-bit ativado")
        except Exception as exc:
            print(f"[Syon/Kaggle] 4-bit indisponível: {exc}")

    load_kwargs: dict = {"device_map": "auto"}
    if bnb_config:
        load_kwargs["quantization_config"] = bnb_config
    else:
        load_kwargs["torch_dtype"] = torch.float16

    model = AutoModelForCausalLM.from_pretrained(base_model, **load_kwargs)

    if use_lora:
        from peft import LoraConfig, get_peft_model

        lora_cfg = LoraConfig(
            r=int(model_cfg.get("lora_r", 16)),
            lora_alpha=int(model_cfg.get("lora_alpha", 32)),
            lora_dropout=float(model_cfg.get("lora_dropout", 0.05)),
            target_modules=list(model_cfg.get("lora_target_modules", ["q_proj", "v_proj"])),
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()

    return tokenizer, model, max_len


class SyonKaggleDataset(torch.utils.data.Dataset):
    def __init__(self, samples, tokenizer, max_length: int):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        encoded = self.tokenizer(
            self.samples[idx].text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        ids = encoded["input_ids"].squeeze(0)
        mask = encoded["attention_mask"].squeeze(0)
        labels = ids.clone()
        labels[mask == 0] = -100
        return {"input_ids": ids, "attention_mask": mask, "labels": labels}


def collate_batch(batch: list[dict]) -> dict:
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}


def train_on_kaggle(config_path: Path, data_dir: Path, max_steps: int | None = None) -> Path:
    from scripts.kaggle.prepare_dataset import prepare
    from training.core.trainer import Trainer, load_training_config
    from training.losses.security_aware import SecurityAwareLoss

    config = load_config(config_path)
    tcfg = config.get("training_params", {})
    steps = max_steps or int(tcfg.get("max_steps", 500))
    batch_size = int(tcfg.get("batch_size", 2))
    accum = int(tcfg.get("gradient_accumulation_steps", 8))
    lr = float(tcfg.get("learning_rate", 2e-4))
    save_every = int(tcfg.get("save_every_steps", 100))
    sec_weight = float(config.get("loss", {}).get("security_aware_weight", 0.15))

    kaggle_input = Path("/kaggle/input")
    prepare(data_dir, kaggle_input if kaggle_input.exists() else None)

    device = detect_device()
    tokenizer, model, max_len = build_model(config)

    trainer_cfg = load_training_config(config_path)
    curator = Trainer(trainer_cfg, data_dir)
    samples = curator.prepare_data()
    print(f"[Syon/Kaggle] Amostras: {len(samples)}")

    loader = DataLoader(
        SyonKaggleDataset(samples, tokenizer, max_len),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_batch,
        drop_last=False,
    )

    batches_per_epoch = max(1, len(loader))
    if accum > batches_per_epoch:
        print(f"[Syon/Kaggle] accum {accum} → {batches_per_epoch} (só {len(samples)} amostras)")
        accum = batches_per_epoch

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=0.01,
    )
    security_loss_fn = SecurityAwareLoss(weight=sec_weight)
    use_cuda = device.type == "cuda"
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)
        autocast_ctx = lambda: torch.amp.autocast("cuda", enabled=use_cuda)
    except AttributeError:
        scaler = torch.cuda.amp.GradScaler(enabled=use_cuda)
        autocast_ctx = lambda: torch.cuda.amp.autocast(enabled=use_cuda)

    output_dir = Path(config.get("output_dir", "/kaggle/working/syon-output"))
    if not str(output_dir).startswith("/kaggle") and is_kaggle():
        output_dir = Path("/kaggle/working/syon-output")
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "training_log.jsonl"

    model.train()
    step = 0
    micro_step = 0
    running_loss = 0.0
    optimizer.zero_grad()

    max_epochs = int(tcfg.get("num_epochs", 50))
    print(
        f"[Syon/Kaggle] Treino: {steps} steps | batch={batch_size} | accum={accum} | "
        f"batches/epoch={batches_per_epoch}"
    )

    epoch = 0
    while step < steps and epoch < max_epochs:
        epoch += 1
        for batch in loader:
            if step >= steps:
                break

            batch = {k: v.to(device) for k, v in batch.items()}

            with autocast_ctx():
                out = model(**batch)
                ce = F.cross_entropy(
                    out.logits.view(-1, out.logits.size(-1)),
                    batch["labels"].view(-1),
                    ignore_index=-100,
                )
                sec = security_loss_fn(out.logits, batch["labels"])
                loss = (ce + sec) / accum

            scaler.scale(loss).backward()
            running_loss += float(loss.item()) * accum
            micro_step += 1

            if micro_step % accum == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                step += 1
                print(f"  step {step}/{steps} | loss={running_loss/accum:.4f} | ce={float(ce):.4f}")
                running_loss = 0.0

                if step % save_every == 0:
                    ckpt = output_dir / f"checkpoint_step_{step}"
                    model.save_pretrained(ckpt)
                    tokenizer.save_pretrained(ckpt)
                    print(f"  saved: {ckpt}")

                with log_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps({"step": step, "loss": float(ce), "security": float(sec)}) + "\n")

        if micro_step % accum != 0 and step < steps:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            step += 1
            print(f"  step {step}/{steps} (flush epoch {epoch}) | loss={running_loss:.4f}")
            running_loss = 0.0
            micro_step = 0

    final_dir = output_dir / "syon-kaggle-lora"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

    summary = {
        "steps_completed": step,
        "epochs_run": epoch,
        "samples": len(samples),
        "gradient_accumulation": accum,
        "base_model": config.get("model_params", {}).get("base_model"),
        "output": str(final_dir),
        "platform": "kaggle",
    }
    (output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[Syon/Kaggle] Concluído → {final_dir} | steps={step} | epochs={epoch}")
    return final_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Syon Kaggle Trainer")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "training/configs/kaggle_config.yaml")
    parser.add_argument("--data-dir", type=Path, default=Path("/kaggle/working/syon/data/raw"))
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args()

    if not is_kaggle() and str(args.data_dir).startswith("/kaggle"):
        args.data_dir = PROJECT_ROOT / "data/raw"

    train_on_kaggle(args.config, args.data_dir, args.max_steps)


if __name__ == "__main__":
    main()