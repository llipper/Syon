"""Testes do pipeline HF de conversação."""

from models.tokenizer.syon_bpe import SyonBPETokenizer
from training.hf.conversation_data import normalize_turns, turns_to_sft_features


def _tiny_tokenizer() -> SyonBPETokenizer:
    return SyonBPETokenizer.train(
        ["<|user|>oi <|assistant|>olá como posso ajudar"],
        vocab_size=256,
        max_length=64,
    )


def test_normalize_turns():
    turns = normalize_turns(
        [
            {"role": "user", "content": "Explique SQL injection"},
            {"role": "assistant", "content": "SQL injection ocorre quando..."},
        ]
    )
    assert turns is not None
    assert turns[0]["role"] == "user"
    assert turns[1]["role"] == "assistant"


def test_sft_masks_assistant_only():
    tok = _tiny_tokenizer()
    turns = [
        {"role": "user", "content": "oi"},
        {"role": "assistant", "content": "olá"},
    ]
    feat = turns_to_sft_features(tok, turns, max_length=32)
    assert feat is not None
    assert len(feat["input_ids"]) == 32
    assert -100 in feat["labels"]
    assert any(l != -100 for l in feat["labels"])
    assert feat["labels"][0] == -100