"""
Bootstrap Syon 3 no Kaggle — código completo + kl/ (pesos + tokenizer).

Uso:
    python scripts/kaggle/kaggle_bootstrap.py --project /kaggle/working/syon
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def is_full_project(path: Path) -> bool:
    has_trainer = (path / "training" / "hf" / "syon3_hf_trainer.py").exists()
    has_arch = (path / "models" / "architecture" / "syon3.py").exists()
    return has_trainer and has_arch


def _path_kind(p: Path) -> str:
    s = str(p).replace("\\", "/")
    if "/kaggle/input/datasets/" in s:
        return "dataset"
    if "/kaggle/input/models/" in s:
        return "model"
    return "other"


def find_project(input_dir: Path) -> Path | None:
    if not input_dir.exists():
        return None

    candidates: list[Path] = []
    for marker in input_dir.rglob("syon3_hf_trainer.py"):
        if marker.parent.name != "hf":
            continue
        root = marker.parent.parent.parent
        if is_full_project(root):
            candidates.append(root)

    if not candidates:
        return None

    def rank(p: Path) -> tuple[int, int, int]:
        kind = _path_kind(p)
        kind_score = {"dataset": 3, "other": 2, "model": 1}[kind]
        has_kl = 1 if (p / "kl" / "tokenizer.json").exists() else 0
        return (kind_score, has_kl, -len(str(p)))

    return max(candidates, key=rank)


def find_kl_source(input_dir: Path, project: Path) -> Path | None:
    """Prioriza Kaggle Model (/kaggle/input) depois kl/ do projeto."""
    project_kl = project / "kl"
    if project_kl.is_dir() and (project_kl / "tokenizer.json").exists():
        return project_kl

    for base in (input_dir,):
        direct = base / "kl"
        if direct.is_dir() and (direct / "tokenizer.json").exists():
            return direct
        for tok in base.rglob("tokenizer.json"):
            parent = tok.parent
            if (parent / "config.json").exists() and (
                (parent / "pytorch_model.bin").exists() or parent.name == "kl"
            ):
                return parent

    return None


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
            "UltrachatBR",
            "dataset-v3-bucket",
            "terminals",
        ),
    )


def copy_kl(src: Path, dest: Path) -> None:
    if src.resolve() == dest.resolve():
        print(f"[bootstrap] kl/ já no lugar: {dest}")
        return
    dest.mkdir(parents=True, exist_ok=True)
    for name in (
        "pytorch_model.bin",
        "config.json",
        "tokenizer.json",
        "syon_model.json",
        "model.safetensors",
    ):
        f = src / name
        if f.exists():
            shutil.copy2(f, dest / name)


def find_conversation_files(input_dir: Path) -> list[Path]:
    names = {"instruct_aira_pt.jsonl", "ultrachat_br.jsonl"}
    return [p for p in input_dir.rglob("*.jsonl") if p.name in names]


def resolve_code_root(project: Path, input_dir: Path) -> Path:
    """Código: git clone em /kaggle/working/syon OU dataset em /kaggle/input."""
    if is_full_project(project):
        print(f"[bootstrap] Código OK (git clone): {project}")
        return project

    src = find_project(input_dir)
    if src is not None:
        print(f"[bootstrap] Código ({_path_kind(src)}): {src}")
        if src.resolve() != project.resolve():
            print(f"[bootstrap] Copiando → {project}")
            copy_tree(src, project)
        return project

    raise FileNotFoundError(
        "Código Syon 3 não encontrado.\n"
        "Rode antes: !git clone https://github.com/llipper/Syon.git /kaggle/working/syon"
    )


def bootstrap(project: Path, input_dir: Path) -> dict:
    report: dict = {"project": str(project), "input": str(input_dir)}

    resolve_code_root(project, input_dir)

    kl_dir = project / "kl"
    # Pesos/tokenizer: Kaggle Model em /kaggle/input (kl/ não vai no GitHub — *.bin no .gitignore)
    kl_src = find_kl_source(input_dir, project)
    if kl_src:
        print(f"[bootstrap] kl/: {kl_src} → {kl_dir}")
        copy_kl(kl_src, kl_dir)
        report["kl_source"] = str(kl_src)
    elif (kl_dir / "tokenizer.json").exists():
        print(f"[bootstrap] kl/ já presente em {kl_dir}")
    else:
        raise FileNotFoundError(
            "tokenizer.json não encontrado.\n"
            "Add Data → Model regyfelipe/syon-3-v1 (ou pasta kl/ com tokenizer + pesos)"
        )

    has_weights = (kl_dir / "pytorch_model.bin").exists()
    report["has_weights"] = has_weights
    if not has_weights:
        print("[bootstrap] AVISO: sem pytorch_model.bin — treino inicia pesos aleatórios")

    conv_dir = project / "data" / "raw" / "conversation"
    conv_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for f in find_conversation_files(input_dir):
        shutil.copy2(f, conv_dir / f.name)
        copied += 1
        print(f"[bootstrap] Conversação: {f.name}")
    report["conversation_files"] = copied
    if copied == 0:
        print("[bootstrap] Sem conversação local — usa API HF datasets (Internet ON)")

    report["kl"] = str(kl_dir)
    (project / "kaggle_bootstrap_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap Syon 3 no Kaggle")
    parser.add_argument("--project", type=Path, default=Path("/kaggle/working/syon"))
    parser.add_argument("--input", type=Path, default=Path("/kaggle/input"))
    args = parser.parse_args()

    report = bootstrap(args.project, args.input)
    print(f"[bootstrap] OK → {args.project}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()