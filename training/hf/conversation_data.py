"""Dataset de conversação para SFT — Hugging Face datasets + máscara de labels."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

from datasets import Dataset, load_dataset

from models.tokenizer.syon_bpe import SyonBPETokenizer
from training.hf.policy import assert_dataset_allowed

ROLE_TAGS = {
    "user": "<|user|>",
    "model": "<|assistant|>",
    "assistant": "<|assistant|>",
}
TRAINABLE_ROLES = {"model", "assistant"}


def normalize_turns(raw_turns: list[dict[str, Any]]) -> list[dict[str, str]] | None:
    turns: list[dict[str, str]] = []
    for t in raw_turns:
        role = str(t.get("role", t.get("from", ""))).strip().lower()
        if role in ("human", "humano"):
            role = "user"
        if role in ("gpt", "bot"):
            role = "model"
        content = str(t.get("content", t.get("value", ""))).strip()
        if role not in ROLE_TAGS or len(content) < 2:
            continue
        turns.append({"role": role, "content": content})
    if len(turns) < 2:
        return None
    if not any(t["role"] == "user" for t in turns) or not any(t["role"] in TRAINABLE_ROLES for t in turns):
        return None
    return turns


def turns_to_sft_features(
    tokenizer: SyonBPETokenizer,
    turns: list[dict[str, str]],
    max_length: int,
) -> dict[str, list[int]] | None:
    """Tokeniza diálogo e mascara labels — treina só respostas do assistente."""
    input_ids = [tokenizer.bos_token_id]
    labels = [-100]

    for turn in turns:
        tag = ROLE_TAGS.get(turn["role"])
        if not tag:
            continue
        content = turn["content"].strip()
        if not content:
            continue

        train_content = turn["role"] in TRAINABLE_ROLES
        for text, train in ((tag, False), (content, train_content)):
            piece = tokenizer.encode(text, add_special=False)
            if not piece:
                continue
            remaining = max_length - len(input_ids)
            if remaining <= 0:
                break
            piece = piece[:remaining]
            input_ids.extend(piece)
            labels.extend(piece if train else [-100] * len(piece))

        if len(input_ids) >= max_length:
            break

    if len(input_ids) < 4:
        return None
    if all(l == -100 for l in labels):
        return None

    if len(input_ids) < max_length:
        input_ids.append(tokenizer.eos_token_id)
        labels.append(-100)
    else:
        input_ids[-1] = tokenizer.eos_token_id
        labels[-1] = -100

    attention_mask = [1] * len(input_ids)
    pad_len = max_length - len(input_ids)
    if pad_len > 0:
        input_ids.extend([tokenizer.pad_token_id] * pad_len)
        labels.extend([-100] * pad_len)
        attention_mask.extend([0] * pad_len)

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
    }


def _iter_local_jsonl(paths: list[Path]) -> Iterator[dict[str, Any]]:
    for path in paths:
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                yield json.loads(line)


def load_conversation_records(
    *,
    local_files: list[Path],
    hf_dataset: str | None = None,
    hf_split: str = "portuguese",
    max_samples: int = 50_000,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    for row in _iter_local_jsonl(local_files):
        turns = row.get("turns")
        if isinstance(turns, list):
            normalized = normalize_turns(turns)
            if normalized:
                records.append({"turns": normalized})
        elif row.get("text"):
            records.append({"text": row["text"]})
        if len(records) >= max_samples:
            return records

    if records or not hf_dataset:
        return records[:max_samples]

    assert_dataset_allowed(hf_dataset)
    print(f"[Syon/HF] API datasets (texto): {hf_dataset} ({hf_split})...")
    ds = load_dataset(hf_dataset, split=hf_split, streaming=True)
    for row in ds:
        turns = normalize_turns(row.get("conversations") or [])
        if turns:
            records.append({"turns": turns})
        if len(records) >= max_samples:
            break
    return records


def build_hf_dataset(
    tokenizer: SyonBPETokenizer,
    records: list[dict[str, Any]],
    max_length: int,
    *,
    progress_every: int = 5000,
) -> Dataset:
    features: list[dict[str, list[int]]] = []
    total = len(records)
    print(f"[Syon/HF] Tokenizando {total} amostras (pode levar alguns minutos)...")

    for i, record in enumerate(records, start=1):
        if "turns" in record:
            item = turns_to_sft_features(tokenizer, record["turns"], max_length)
        else:
            text = str(record.get("text", "")).strip()
            if not text:
                continue
            ids = tokenizer.encode(text, add_special=True)
            if len(ids) > max_length:
                ids = ids[:max_length]
                ids[-1] = tokenizer.eos_token_id
            labels = ids.copy()
            mask_len = max_length - len(ids)
            attention_mask = [1] * len(ids) + [0] * mask_len
            ids = ids + [tokenizer.pad_token_id] * mask_len
            labels = labels + [-100] * mask_len
            item = {"input_ids": ids, "labels": labels, "attention_mask": attention_mask}
        if item:
            features.append(item)

        if i % progress_every == 0 or i == total:
            pct = 100 * i / total
            print(f"[Syon/HF] Tokenizados {i}/{total} ({pct:.0f}%) — válidos: {len(features)}")

    if not features:
        raise ValueError("Nenhuma amostra de conversação válida para treino HF")

    print(f"[Syon/HF] Dataset pronto: {len(features)} amostras")
    return Dataset.from_list(features)


def load_mixed_conversation_dataset(
    tokenizer: SyonBPETokenizer,
    *,
    conversation_dir: Path,
    max_length: int,
    max_samples: int,
    hf_dataset: str | None = "nicholasKluge/instruct-aira-dataset-v3",
    hf_split: str = "portuguese",
    val_ratio: float = 0.02,
) -> tuple[Dataset, Dataset | None]:
    local_files = [
        conversation_dir / "instruct_aira_pt.jsonl",
        conversation_dir / "ultrachat_br.jsonl",
    ]
    records = load_conversation_records(
        local_files=local_files,
        hf_dataset=hf_dataset,
        hf_split=hf_split,
        max_samples=max_samples,
    )
    print(f"[Syon/HF] {len(records)} diálogos carregados")

    dataset = build_hf_dataset(tokenizer, records, max_length)
    if val_ratio <= 0 or len(dataset) < 50:
        return dataset, None

    split = dataset.train_test_split(test_size=val_ratio, seed=42)
    return split["train"], split["test"]