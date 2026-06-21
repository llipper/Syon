"""Treino COMPLETO Syon+Gemma — 3 fases com raciocínio (QLoRA)."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

PROJECT = Path(__file__).resolve().parents[2]
for p in (str(PROJECT), str(PROJECT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")


def resolve_gemma_path(cfg: dict) -> str | Path:
    """Usa base_model do config (HF id ou pasta local) ou busca em /kaggle/input."""
    base = cfg.get("model_params", {}).get("base_model", "")
    if base and (str(base).startswith("google/") or Path(base).exists()):
        return base

    input_dir = Path("/kaggle/input")
    best: list[tuple[int, Path]] = []
    for cfg_path in input_dir.rglob("config.json"):
        if "gemma" not in str(cfg_path).lower():
            continue
        d = cfg_path.parent
        if list(d.glob("*.safetensors")) or (d / "model.safetensors").exists():
            ver = max((int(x) for x in d.parts if x.isdigit()), default=0)
            best.append((ver, d))
    if not best:
        return "google/gemma-2-2b-it"
    for _, p in sorted(best):
        if "2b" in str(p).lower():
            return p
    return sorted(best)[-1][1]


def load_samples(sft_dir: Path) -> list[dict]:
    path = sft_dir / "gemma_sft_train.jsonl"
    records: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        records.append({
            "text": obj["text"],
            "domain": str(obj.get("domain", "")),
            "conv_type": str(obj.get("conv_type", "")),
        })
    return records


def filter_phase(samples: list[dict], phase_cfg: dict) -> list[str]:
    domains = phase_cfg.get("focus_domains") or []
    if not domains:
        return [s["text"] for s in samples]

    keys_map = {
        "programming": ("python", "rust", "go", "java", "código", "code", "api", "function", "class", "refactor", "git", "test"),
        "cybersecurity": ("cve", "cwe", "xss", "sql", "owasp", "injection", "segur", "vulner", "auth", "cripto", "malware", "forensic", "threat"),
        "complementary": ("arquitet", "microservice", "devops", "cloud", "sre", "observ", "cap", "distributed"),
        "training_quality": ("review", "audit", "senior", "master", "qualidade", "instruction"),
        "conversation": ("<start_of_turn>user", "convers", "olá", "bom dia", "obrigado", "entendi", "pode me", "como você"),
    }
    filtered: list[str] = []
    for s in samples:
        domain = s.get("domain", "")
        if domain and domain in domains:
            filtered.append(s["text"])
            continue
        low = s["text"].lower()
        for d in domains:
            if any(k in low for k in keys_map.get(d, (d,))):
                filtered.append(s["text"])
                break
    return filtered if filtered else [s["text"] for s in samples]


class TextDS(torch.utils.data.Dataset):
    def __init__(self, texts: list[str], tok, max_len: int):
        self.texts, self.tok, self.max_len = texts, tok, max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, i: int) -> dict:
        enc = self.tok(self.texts[i], truncation=True, max_length=self.max_len,
                       padding="max_length", return_tensors="pt")
        ids = enc["input_ids"].squeeze(0)
        mask = enc["attention_mask"].squeeze(0)
        labels = ids.clone()
        labels[mask == 0] = -100
        return {"input_ids": ids, "attention_mask": mask, "labels": labels}


def collate(batch: list[dict]) -> dict:
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}


def build_model(gemma_path: Path | str, cfg: dict):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    mp = cfg["model_params"]
    tok = AutoTokenizer.from_pretrained(str(gemma_path))
    tok.pad_token = tok.eos_token

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(gemma_path), quantization_config=bnb, device_map={"": 0}, dtype=torch.float16,
    )
    model = get_peft_model(
        model,
        LoraConfig(
            r=int(mp["lora_r"]), lora_alpha=int(mp["lora_alpha"]),
            lora_dropout=float(mp["lora_dropout"]), task_type="CAUSAL_LM",
            target_modules=list(mp["lora_target_modules"]),
        ),
    )
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    return tok, model


def train_phase(model, tok, samples, phase_cfg, train_cfg, device, start_step: int = 0) -> int:
    max_len = int(train_cfg.get("model_params", {}).get("max_seq_length", 768))
    bs = int(train_cfg["training_params"]["batch_size"])
    accum = int(train_cfg["training_params"]["gradient_accumulation_steps"])
    max_steps = int(phase_cfg["max_steps"])
    lr = float(phase_cfg["learning_rate"])
    save_every = int(train_cfg["training_params"]["save_every_steps"])

    loader = DataLoader(TextDS(samples, tok, max_len), batch_size=bs, shuffle=True, collate_fn=collate)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)

    model.train()
    step, micro = start_step, 0
    print(f"\n{'='*60}\nFASE: {phase_cfg['name']} | {phase_cfg.get('description','')} | steps={max_steps}\n{'='*60}")

    for epoch in range(int(train_cfg["training_params"]["num_epochs"])):
        for batch in loader:
            if step >= max_steps:
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss / accum
            loss.backward()
            micro += 1
            if micro % accum == 0:
                opt.step()
                opt.zero_grad()
                step += 1
                if step % 20 == 0 or step == 1:
                    print(f"  [{phase_cfg['name']}] step {step}/{max_steps} loss={loss.item()*accum:.4f}")
                if step % save_every == 0:
                    ckpt = Path(train_cfg["output_dir"]) / phase_cfg["name"] / f"step_{step}"
                    model.save_pretrained(ckpt)
                    tok.save_pretrained(ckpt)
        if step >= max_steps:
            break
    return step


def find_kaggle_input(name: str) -> Path | None:
    base = Path("/kaggle/input")
    if not base.exists():
        return None
    for p in base.rglob(name):
        if p.is_file() or p.is_dir():
            return p.parent if p.is_file() else p
    return None


def stage_kaggle_data(cfg: dict) -> None:
    """Copia datasets pre-processados de /kaggle/input para /kaggle/working."""
    import shutil

    data_cfg = cfg.get("data", {})
    raw = Path(data_cfg["raw_dir"])
    sft_dir = Path(data_cfg["sft_dir"])
    sft_train = sft_dir / "gemma_sft_train.jsonl"

    conv_in = data_cfg.get("conversation_input")
    if conv_in:
        src = Path(conv_in)
    else:
        src = find_kaggle_input("instruct_aira_pt.jsonl")
        if src:
            src = src if (src / "instruct_aira_pt.jsonl").exists() else src.parent
    if src and src.exists():
        dst = raw / "conversation"
        dst.mkdir(parents=True, exist_ok=True)
        for f in src.glob("*.jsonl"):
            shutil.copy2(f, dst / f.name)
        print(f"[Syon/Gemma-Full] Conversacao copiada de {src} -> {dst}")

    prebuilt = data_cfg.get("prebuilt_sft_input")
    if prebuilt:
        src_sft = Path(prebuilt)
    else:
        found = find_kaggle_input("gemma_sft_train.jsonl")
        src_sft = found if found and found.name == "gemma_sft_train.jsonl" else (found / "gemma_sft_train.jsonl").parent if found else None
        if src_sft and not (src_sft / "gemma_sft_train.jsonl").exists() and (src_sft.parent / "gemma_sft_train.jsonl").exists():
            src_sft = src_sft.parent
    if src_sft and (Path(src_sft) / "gemma_sft_train.jsonl").exists():
        sft_dir.mkdir(parents=True, exist_ok=True)
        for f in Path(src_sft).glob("*"):
            if f.is_file():
                shutil.copy2(f, sft_dir / f.name)
        print(f"[Syon/Gemma-Full] SFT pre-built copiado de {src_sft} -> {sft_dir}")
        return

    if sft_train.exists() and sft_train.stat().st_size > 1_000_000:
        print(f"[Syon/Gemma-Full] SFT ja existe: {sft_train}")
        return


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PROJECT / "training/configs/syon_gemma_full_kaggle.yaml")
    parser.add_argument("--skip-sft-build", action="store_true")
    args = parser.parse_args()

    with args.config.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    gemma = resolve_gemma_path(cfg)
    cfg["model_params"]["base_model"] = str(gemma)
    print(f"[Syon/Gemma-Full] Base: {gemma}")

    stage_kaggle_data(cfg)

    sft_dir = Path(cfg["data"]["sft_dir"])
    sft_train = sft_dir / "gemma_sft_train.jsonl"
    skip_build = args.skip_sft_build or cfg["data"].get("skip_sft_build", False)

    if not skip_build and not (sft_train.exists() and sft_train.stat().st_size > 1_000_000):
        from scripts.data.build_gemma_sft_dataset import build as build_sft

        raw = Path(cfg["data"]["raw_dir"])
        build_sft(
            sft_dir,
            raw,
            ensure_curriculum=True,
            min_curriculum=int(cfg["data"].get("min_curriculum", 10000)),
            ensure_conversation=not cfg["data"].get("no_conversation_import", False),
            min_conversation=int(cfg["data"].get("min_conversation", 5000)),
        )
    else:
        print(f"[Syon/Gemma-Full] Usando SFT existente: {sft_train}")

    samples = load_samples(sft_dir)
    device = torch.device("cuda:0")
    tok, model = build_model(Path(gemma) if Path(str(gemma)).exists() else gemma, cfg)  # type: ignore[arg-type]
    model.print_trainable_parameters()

    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    results = []

    for phase in cfg["phases"]:
        phase_samples = filter_phase(samples, phase)
        random.shuffle(phase_samples)
        steps = train_phase(model, tok, phase_samples, phase, cfg, device)
        ckpt = out / phase["name"] / "best"
        ckpt.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(ckpt)
        tok.save_pretrained(ckpt)
        results.append({"phase": phase["name"], "steps": steps})

    final = out / "syon-gemma-full"
    final.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final)
    tok.save_pretrained(final)
    (out / "training_summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n[Syon/Gemma-Full] ✓ Concluído → {final}")


if __name__ == "__main__":
    main()