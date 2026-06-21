"""Prepara dataset JSONL para treino no Kaggle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

SAMPLE_DATA: dict[str, dict[str, list[dict]]] = {
    "programming": {
        "github_repositories": [
            {
                "text": (
                    "def validate_email(email: str) -> bool:\n"
                    "    import re\n"
                    "    pattern = r'^[\\w.+-]+@[\\w-]+\\.[a-zA-Z]{2,}$'\n"
                    "    return bool(re.match(pattern, email))\n"
                ),
                "metadata": {"language": "python", "source": "github"},
            },
            {
                "text": (
                    "async function fetchUser(id: string): Promise<User> {\n"
                    "  const res = await fetch(`/api/users/${id}`);\n"
                    "  if (!res.ok) throw new Error('Not found');\n"
                    "  return res.json();\n"
                    "}\n"
                ),
                "metadata": {"language": "typescript", "source": "github"},
            },
        ],
        "stack_overflow_qa": [
            {
                "text": (
                    "Q: Como iterar um dict em Python?\n"
                    "A: Use for key, value in d.items(): para chave e valor.\n"
                ),
                "metadata": {"language": "python", "source": "stackoverflow"},
            },
        ],
        "official_documentation": [
            {
                "text": (
                    "Rust ownership: cada valor tem um owner. "
                    "Quando o owner sai de escopo, o valor é liberado.\n"
                ),
                "metadata": {"language": "rust", "source": "docs"},
            },
        ],
    },
    "cybersecurity": {
        "cve_cwe_cvss_documents": [
            {
                "text": (
                    "CWE-502: Deserialização de dados não confiáveis. "
                    "Nunca use pickle.loads() em entrada do usuário.\n"
                ),
                "metadata": {"cwe": "CWE-502", "severity": "high"},
            },
        ],
        "owasp_top_10": [
            {
                "text": (
                    "A03:2021 Injection — Use prepared statements e ORM "
                    "com parâmetros vinculados. Nunca concatene SQL.\n"
                ),
                "metadata": {"owasp": "A03:2021-Injection"},
            },
        ],
        "vulnerability_reports": [
            {
                "text": (
                    "Vulnerabilidade: API key hardcoded. "
                    "Remediar: usar variáveis de ambiente ou vault.\n"
                ),
                "metadata": {"cwe": "CWE-798", "severity": "medium"},
            },
        ],
    },
    "complementary": {
        "cryptography_mathematics": [
            {
                "text": "AES-256-GCM: cifra autenticada. Use nonce único por mensagem.\n",
                "metadata": {"topic": "cryptography"},
            },
        ],
    },
    "training_quality": {
        "synthetic_ia_generated": [
            {
                "text": (
                    "<|user|>Implemente JWT seguro\n"
                    "<|assistant|>Use HS256 com segredo forte, exp curto, "
                    "e valide iss/aud.\n"
                ),
                "metadata": {"type": "instruction"},
            },
        ],
    },
}


def write_jsonl(path: Path, records: list[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return len(records)


def copy_kaggle_input(input_dir: Path, output_dir: Path) -> int:
    """Copia JSONL de /kaggle/input/<dataset>/ para data/raw."""
    if not input_dir.exists():
        return 0
    count = 0
    for jsonl in input_dir.rglob("*.jsonl"):
        rel = jsonl.relative_to(input_dir)
        dest = output_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(jsonl.read_text(encoding="utf-8"), encoding="utf-8")
        count += 1
    return count


def prepare(data_dir: Path, kaggle_input: Path | None = None, max_per_source: int = 500) -> dict:
    stats: dict[str, int] = {"files": 0, "samples": 0, "from_kaggle_input": 0}

    if kaggle_input and kaggle_input.exists():
        stats["from_kaggle_input"] = copy_kaggle_input(kaggle_input, data_dir)

    for domain, sources in SAMPLE_DATA.items():
        for source, records in sources.items():
            path = data_dir / domain / f"{source}.jsonl"
            if path.exists():
                continue
            limited = records[:max_per_source]
            n = write_jsonl(path, limited)
            stats["files"] += 1
            stats["samples"] += n

    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("/kaggle/working/syon/data/raw"))
    parser.add_argument("--kaggle-input", type=Path, default=Path("/kaggle/input"))
    parser.add_argument("--max-per-source", type=int, default=500)
    args = parser.parse_args()

    stats = prepare(args.data_dir, args.kaggle_input, args.max_per_source)
    print(f"[Syon/Kaggle] Dataset pronto: {stats}")


if __name__ == "__main__":
    main()