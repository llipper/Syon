"""
Bootstrap Syon no Kaggle — código + pesos Syon 3 + dados de conversação.

Uso:
    python scripts/kaggle/kaggle_bootstrap.py
    python scripts/kaggle/kaggle_bootstrap.py --project /kaggle/working/syon
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


def is_project_root(path: Path) -> bool:
    return (path / "training" / "hf" / "syon3_hf_trainer.py").exists() or (
        path / "models" / "architecture" / "syon3.py"
    ).exists()


def find_project(input_dir: Path) -> Path | None:
    if not input_dir.exists():
        return None
    for marker in input_dir.rglob("syon3_hf_trainer.py"):
        if marker.parent.name == "hf" and marker.parent.parent.name == "training":
            root = marker.parent.parent.parent
            if is_project_root(root):
                return root
    for marker in input_dir.rglob("syon3.py"):
        if marker.parent.name == "architecture":
            root = marker.parent.parent.parent
            if is_project_root(root):
                return root
    return None


def find_checkpoint(input_dir: Path) -> Path | None:
    candidates: list[Path] = []
    for weights in input_dir.rglob("pytorch_model.bin"):
        parent = weights.parent
        if (parent / "config.json").exists() and (parent / "tokenizer.json").exists():
            candidates.append(parent)

    if not candidates:
        return None

    def version_key(p: Path) -> int:
        for part in reversed(p.parts):
            if part.isdigit():
                return int(part)
        return 0

    return max(candidates, key=version_key)


def copy_tree(src: Path, dest: Path) -> None:
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(
        src,
        dest,
        ignore=shutil.ignore_patterns(
            ".git",
            "__pycache__",
            "*.pyc",
            ".pytest_cache",
            "node_modules",
            "UltrachatBR",
            "dataset-v3-bucket",
        ),
    )


def copy_checkpoint(src: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for name in ("pytorch_model.bin", "config.json", "tokenizer.json", "syon_model.json"):
        f = src / name
        if f.exists():
            shutil.copy2(f, dest / name)


def find_conversation_files(input_dir: Path) -> list[Path]:
    names = {"instruct_aira_pt.jsonl", "ultrachat_br.jsonl"}
    found: list[Path] = []
    for path in input_dir.rglob("*.jsonl"):
        if path.name in names:
            found.append(path)
    return found


def copy_conversation_data(input_dir: Path, dest_dir: Path) -> int:
    dest_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for src in find_conversation_files(input_dir):
        shutil.copy2(src, dest_dir / src.name)
        copied += 1
        print(f"[bootstrap] Conversação: {src.name}")
    return copied


def bootstrap(project: Path, input_dir: Path) -> dict:
    report: dict = {"project": str(project), "input": str(input_dir)}

    src = find_project(input_dir)
    if src is None:
        if is_project_root(Path.cwd()):
            src = Path.cwd()
            print(f"[bootstrap] Usando código atual: {src}")
        else:
            raise FileNotFoundError(
                "Código Syon não encontrado em /kaggle/input.\n"
                "Add Data → Dataset com código Syon 3 (ex: regyfelipe/syon-3-code)"
            )
    else:
        print(f"[bootstrap] Código: {src} → {project}")
        copy_tree(src, project)

    ckpt = find_checkpoint(input_dir)
    kl_dir = project / "kl"
    if ckpt:
        print(f"[bootstrap] Pesos Syon 3: {ckpt} → {kl_dir}")
        copy_checkpoint(ckpt, kl_dir)
        report["checkpoint"] = str(ckpt)
    elif (project / "kl" / "pytorch_model.bin").exists():
        print(f"[bootstrap] Pesos já em {kl_dir}")
    else:
        raise FileNotFoundError(
            "Pesos Syon 3 não encontrados.\n"
            "Add Data → Model regyfelipe/syon-3 (pesos) ou pasta kl/ com pytorch_model.bin"
        )

    conv_dir = project / "data" / "raw" / "conversation"
    n = copy_conversation_data(input_dir, conv_dir)
    report["conversation_files"] = n
    if n == 0:
        print("[bootstrap] Sem dados locais — treino usará Hugging Face Hub (Internet ON)")

    report["kl"] = str(kl_dir)
    report["conversation_dir"] = str(conv_dir)
    (project / "kaggle_bootstrap_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap Syon no Kaggle")
    parser.add_argument("--project", type=Path, default=Path("/kaggle/working/syon"))
    parser.add_argument("--input", type=Path, default=Path("/kaggle/input"))
    args = parser.parse_args()

    report = bootstrap(args.project, args.input)
    print(f"[bootstrap] OK → {args.project}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()