"""Testes da arquitetura Syon 3 — modelo do zero."""

import torch

from models.architecture.config import SyonModelConfig
from models.architecture.syon3 import Syon3
from models.tokenizer.syon_bpe import SyonBPETokenizer


def test_syon3_forward():
    cfg = SyonModelConfig(vocab_size=256, hidden_size=64, num_layers=2, num_heads=4, intermediate_size=128, max_seq_length=32)
    model = Syon3(cfg)
    ids = torch.randint(0, 256, (2, 16))
    mask = torch.ones(2, 16, dtype=torch.long)
    labels = ids.clone()
    out = model(input_ids=ids, attention_mask=mask, labels=labels)
    assert out.logits.shape == (2, 16, 256)
    assert out.loss is not None


def test_bpe_train_encode():
    texts = ["def secure_login(user): pass", "CVE-2024-1234 buffer overflow"]
    tok = SyonBPETokenizer.train(texts, vocab_size=512, max_length=64)
    enc = tok("secure code example", return_tensors="pt")
    assert enc["input_ids"].shape[1] == 64
    assert tok.vocab_size >= 9