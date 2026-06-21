"""
Dataset SFT FULL Syon + Gemma — curriculum técnico + conversação REAL em PT.

Conversação vem de datasets HuggingFace importados (data/raw/conversation/),
não de diálogos escritos no código.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DOMAIN_THINK = {
    "programming": (
        "1) Identificar linguagem, padrão e requisitos.\n"
        "2) Analisar edge cases, performance e testabilidade.\n"
        "3) Aplicar boas práticas (SOLID, clean code, testes).\n"
        "4) Formular solução clara com exemplos quando útil."
    ),
    "cybersecurity": (
        "1) Identificar ativo, ameaça e vetor de ataque.\n"
        "2) Avaliar impacto (CIA) e superfície exposta.\n"
        "3) Mapear controles: prevenção, detecção, resposta.\n"
        "4) Recomendar mitigação priorizada e verificável."
    ),
    "complementary": (
        "1) Contextualizar arquitetura/sistema/distribuído.\n"
        "2) Trade-offs (CAP, consistência, latência, custo).\n"
        "3) Padrões e anti-patterns relevantes.\n"
        "4) Recomendação prática para produção."
    ),
    "training_quality": (
        "1) Decompor o problema em passos verificáveis.\n"
        "2) Critérios de qualidade e segurança.\n"
        "3) Validação e revisão por pares.\n"
        "4) Resposta estruturada e acionável."
    ),
    "default": (
        "1) Entender o enunciado e termos técnicos.\n"
        "2) Raciocinar passo a passo.\n"
        "3) Conectar com segurança e boas práticas.\n"
        "4) Responder de forma completa em português."
    ),
}

USER_PREFIX = {
    "programming": "Como especialista em programação, analise e explique:\n",
    "cybersecurity": "Como especialista em cybersegurança, analise e explique:\n",
    "complementary": "Como arquiteto de software, analise e explique:\n",
    "training_quality": "Como revisor sênior, analise e explique:\n",
    "default": "Analise com raciocínio passo a passo:\n",
}


def format_gemma(user: str, think: str, answer: str) -> str:
    return (
        f"<start_of_turn>user\n{user.strip()}<end_of_turn>\n"
        f"<start_of_turn>model\n<think>\n{think.strip()}\n</think>\n{answer.strip()}<end_of_turn>\n"
    )


def format_gemma_chat(turns: list[tuple[str, str]]) -> str:
    parts = [
        f"<start_of_turn>{role}\n{content.strip()}<end_of_turn>"
        for role, content in turns
    ]
    return "\n".join(parts) + "\n"


def turns_to_gemma(turns: list[dict[str, str]]) -> str:
    pairs = [(t["role"], t["content"]) for t in turns if t.get("role") in ("user", "model")]
    return format_gemma_chat(pairs)


def domain_from_path(path: Path) -> str:
    parts = path.parts
    for d in ("conversation", "programming", "cybersecurity", "complementary", "training_quality"):
        if d in parts:
            return d
    return "default"


def text_to_sft(text: str, domain: str) -> str:
    think = DOMAIN_THINK.get(domain, DOMAIN_THINK["default"])
    prefix = USER_PREFIX.get(domain, USER_PREFIX["default"])
    return format_gemma(prefix + text, think, text)


def load_all_raw(raw_dir: Path) -> list[dict]:
    """Carrega jsonl de data/raw — texto técnico ou turnos de conversação real."""
    items: list[dict] = []
    for path in sorted(raw_dir.rglob("*.jsonl")):
        domain = domain_from_path(path)
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            if domain == "conversation" and isinstance(obj.get("turns"), list):
                turns = obj["turns"]
                if len(turns) >= 2:
                    items.append({
                        "domain": "conversation",
                        "turns": turns,
                        "metadata": obj.get("metadata", {}),
                    })
                continue

            text = str(obj.get("text", obj.get("content", ""))).strip()
            if len(text) > 30:
                items.append({"domain": domain, "text": text})
    return items


def ensure_conversation_data(raw_dir: Path, min_dialogues: int) -> None:
    conv_dir = raw_dir / "conversation"
    manifest = conv_dir / "import_manifest.json"
    ok = False
    if manifest.exists():
        try:
            info = json.loads(manifest.read_text(encoding="utf-8"))
            if int(info.get("total_dialogues", 0)) >= min_dialogues:
                ok = True
                print(f"[Syon/Gemma-FULL] Conversação real OK: {info['total_dialogues']} diálogos")
        except (json.JSONDecodeError, KeyError, TypeError):
            pass
    if not ok:
        print(f"[Syon/Gemma-FULL] Importando conversação PT real ({min_dialogues}+ diálogos)...")
        from scripts.data.import_conversation_datasets import import_all

        aira = min(min_dialogues, 50_000)
        ultra = max(min_dialogues - aira, 5_000)
        import_all(conv_dir, aira_limit=aira, ultrachat_limit=ultra)


def build(
    output: Path,
    raw_dir: Path,
    *,
    ensure_curriculum: bool = True,
    min_curriculum: int = 10000,
    ensure_conversation: bool = True,
    min_conversation: int = 5000,
) -> int:
    output.mkdir(parents=True, exist_ok=True)

    if ensure_curriculum:
        manifest = raw_dir.parent / "curriculum" / "build_manifest.json"
        need_build = True
        if manifest.exists():
            try:
                info = json.loads(manifest.read_text(encoding="utf-8"))
                if int(info.get("total_samples", 0)) >= min_curriculum:
                    need_build = False
                    print(f"[Syon/Gemma-FULL] Curriculum OK: {info['total_samples']} amostras")
            except (json.JSONDecodeError, KeyError):
                pass
        if need_build:
            print(f"[Syon/Gemma-FULL] Gerando curriculum FULL ({min_curriculum}+)...")
            from scripts.data.build_master_curriculum import build as build_curr

            build_curr(raw_dir, min_samples=min_curriculum, augment=5)

    if ensure_conversation:
        ensure_conversation_data(raw_dir, min_conversation)

    items = load_all_raw(raw_dir)
    if not items:
        raise FileNotFoundError(f"Sem dados em {raw_dir}")

    records: list[dict] = []
    for item in items:
        if "turns" in item:
            records.append({
                "text": turns_to_gemma(item["turns"]),
                "domain": "conversation",
                "conv_type": "multi_turn",
                "source": item.get("metadata", {}).get("source", ""),
            })
        else:
            records.append({
                "text": text_to_sft(item["text"], item["domain"]),
                "domain": item["domain"],
                "full": True,
            })

    train_path = output / "gemma_sft_train.jsonl"
    with train_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    by_domain: dict[str, int] = {}
    conv_sources: dict[str, int] = {}
    for rec in records:
        d = str(rec.get("domain", "unknown"))
        by_domain[d] = by_domain.get(d, 0) + 1
        if d == "conversation":
            src = str(rec.get("source", "unknown"))
            conv_sources[src] = conv_sources.get(src, 0) + 1

    stats = {
        "total": len(records),
        "domains": by_domain,
        "conversation_sources": conv_sources,
        "with_think": sum("<think>" in r["text"] for r in records),
        "real_conversation_data": True,
        "full_text": True,
        "no_sample_cap": True,
    }
    (output / "gemma_sft_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"[Syon/Gemma-FULL] ✓ {len(records)} amostras SFT → {train_path}")
    print(f"  conversation (datasets reais): {by_domain.get('conversation', 0)}")
    for d, n in sorted(by_domain.items()):
        print(f"  {d}: {n}")
    return len(records)


def main() -> None:
    parser = argparse.ArgumentParser(description="Dataset Gemma SFT FULL")
    parser.add_argument("--output", type=Path, default=Path("data/processed/gemma_sft"))
    parser.add_argument("--raw", type=Path, default=Path("data/raw"))
    parser.add_argument("--min-curriculum", type=int, default=10000)
    parser.add_argument("--min-conversation", type=int, default=5000)
    parser.add_argument("--no-rebuild", action="store_true")
    parser.add_argument("--no-conversation-import", action="store_true")
    args = parser.parse_args()
    build(
        args.output,
        args.raw,
        ensure_curriculum=not args.no_rebuild,
        min_curriculum=args.min_curriculum,
        ensure_conversation=not args.no_conversation_import,
        min_conversation=args.min_conversation,
    )


if __name__ == "__main__":
    main()