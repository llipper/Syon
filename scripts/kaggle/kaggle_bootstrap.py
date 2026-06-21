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


KL_FILES = (
    "pytorch_model.bin",
    "config.json",
    "tokenizer.json",
    "syon_model.json",
    "model.safetensors",
)


def _version_key(path: Path) -> int:
    for part in reversed(path.parts):
        if part.isdigit():
            return int(part)
    return 0


def find_checkpoints(base: Path) -> list[Path]:
    if not base.exists():
        return []
    found: list[Path] = []
    seen: set[Path] = set()
    for marker in ("pytorch_model.bin", "tokenizer.json", "model.safetensors"):
        for f in base.rglob(marker):
            parent = f.parent
            if parent in seen:
                continue
            if (parent / "config.json").exists() or marker == "tokenizer.json":
                seen.add(parent)
                found.append(parent)
    return found


def find_best_checkpoint(input_dir: Path) -> Path | None:
    """Busca pesos no Kaggle Model (/kaggle/input/models/...)."""
    candidates = find_checkpoints(input_dir)
    if not candidates:
        return None

    def rank(p: Path) -> tuple[int, int, int]:
        has_bin = 1 if (p / "pytorch_model.bin").exists() else 0
        in_models = 1 if "/kaggle/input/models/" in str(p).replace("\\", "/") else 0
        return (has_bin, in_models, _version_key(p))

    return max(candidates, key=rank)


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


def merge_into_kl(kl_dir: Path, *sources: Path | None) -> list[str]:
    """Mescla tokenizer/pesos de várias fontes em project/kl/."""
    kl_dir.mkdir(parents=True, exist_ok=True)
    used: list[str] = []

    for src in sources:
        if src is None or not src.exists():
            continue
        if src.resolve() == kl_dir.resolve():
            used.append(str(src))
            print(f"[bootstrap] kl/ local: {kl_dir}")
            continue
        print(f"[bootstrap] Mesclando → kl/: {src}")
        used.append(str(src))
        for name in KL_FILES:
            f = src / name
            if f.exists():
                shutil.copy2(f, kl_dir / name)

    return used


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
    model_ckpt = find_best_checkpoint(input_dir)
    project_kl = kl_dir if (kl_dir / "tokenizer.json").exists() else None
    # Model Kaggle primeiro (pesos), depois kl/ residual do working
    sources = merge_into_kl(kl_dir, model_ckpt, project_kl)
    report["kl_sources"] = sources

    if not (kl_dir / "tokenizer.json").exists():
        raise FileNotFoundError(
            "tokenizer.json não encontrado.\n"
            "Add Data → Model regyfelipe/syon-3-v1 (deve ter pytorch_model.bin + tokenizer.json)"
        )

    has_weights = (kl_dir / "pytorch_model.bin").exists()
    report["has_weights"] = has_weights
    if not has_weights:
        bins = list(input_dir.rglob("pytorch_model.bin")) if input_dir.exists() else []
        print("[bootstrap] AVISO: sem pytorch_model.bin no kl/")
        print(f"[bootstrap] Arquivos .bin em /kaggle/input: {len(bins)}")
        for b in bins[:5]:
            print(f"  - {b}")
        print("[bootstrap] Treino usará pesos aleatórios se não achar pesos")

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