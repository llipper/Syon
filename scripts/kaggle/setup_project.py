"""
Localiza e copia o projeto Syon de /kaggle/input para /kaggle/working/syon.

Caminhos suportados (Kaggle 2024+):
  /kaggle/input/datasets/<user>/<dataset>/Syon/
  /kaggle/input/<dataset>/Syon/
  /kaggle/input/<dataset>/training/  (raiz direta)
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


def is_project_root(path: Path) -> bool:
    """Raiz do Syon — exige training/ de topo + pyproject.toml ou scripts/kaggle/."""
    if not (path / "training").is_dir():
        return False
    if (path / "pyproject.toml").exists():
        return True
    if (path / "models" / "architecture" / "syon3.py").exists():
        return True
    if (path / "scripts" / "kaggle" / "run_scratch_kaggle.py").exists():
        return True
    if (path / "scripts" / "kaggle" / "kaggle_train.py").exists():
        return True
    if (path / "requirements.txt").exists() and (path / "scripts").is_dir():
        return True
    return False


def find_by_syon3(input_dir: Path) -> Path | None:
    for marker in input_dir.rglob("syon3.py"):
        if marker.parent.name == "architecture" and marker.parent.parent.name == "models":
            root = marker.parent.parent.parent
            if is_project_root(root):
                return root
    return None


def find_by_kaggle_train(input_dir: Path) -> Path | None:
    """Localiza pela árvore: .../scripts/kaggle/kaggle_train.py"""
    for script in input_dir.rglob("kaggle_train.py"):
        if script.parent.name != "kaggle":
            continue
        scripts_dir = script.parent.parent
        if scripts_dir.name != "scripts":
            continue
        root = scripts_dir.parent
        if is_project_root(root):
            return root
    return None


def find_by_pyproject(input_dir: Path) -> Path | None:
    for toml in input_dir.rglob("pyproject.toml"):
        root = toml.parent
        if is_project_root(root):
            return root
    return None


def find_kaggle_datasets_path(input_dir: Path, dataset_slug: str | None = None) -> Path | None:
    """
    Busca em /kaggle/input/datasets/<user>/<slug>/Syon/
    dataset_slug ex: 'syon-project' ou 'regyfelipe/syon-project'
    """
    datasets_dir = input_dir / "datasets"
    if not datasets_dir.is_dir():
        return None

    if dataset_slug:
        parts = Path(dataset_slug).parts
        if len(parts) == 2:
            candidate = datasets_dir / parts[0] / parts[1] / "Syon"
        else:
            for user_dir in datasets_dir.iterdir():
                candidate = user_dir / dataset_slug / "Syon"
                if candidate.is_dir():
                    return candidate if is_project_root(candidate) else None
            candidate = datasets_dir / dataset_slug / "Syon"
        if candidate.is_dir() and is_project_root(candidate):
            return candidate

    for syon_dir in datasets_dir.rglob("Syon"):
        if syon_dir.is_dir() and is_project_root(syon_dir):
            return syon_dir

    return None


def find_project_root(input_dir: Path) -> Path | None:
    if not input_dir.exists():
        return None

    if is_project_root(input_dir):
        return input_dir

    for finder in (find_by_syon3, find_kaggle_datasets_path, find_by_kaggle_train, find_by_pyproject):
        if finder == find_kaggle_datasets_path:
            found = finder(input_dir, None)
        else:
            found = finder(input_dir)
        if found:
            return found

    return None


def list_input_contents(input_dir: Path) -> list[str]:
    lines: list[str] = []
    if not input_dir.exists():
        return ["(pasta /kaggle/input não existe)"]
    for item in sorted(input_dir.iterdir()):
        if item.is_dir():
            children = list(item.iterdir())[:8]
            preview = ", ".join(c.name for c in children)
            lines.append(f"  {item.name}/ → {preview}")
        else:
            lines.append(f"  {item.name}")
    markers = list(input_dir.rglob("pyproject.toml"))[:3]
    for m in markers:
        lines.append(f"  [marker] {m}")
    return lines


def setup(
    input_dir: Path,
    dest: Path,
    dataset_slug: str | None = None,
    src_override: Path | None = None,
) -> Path:
    print("[setup] Conteúdo de /kaggle/input:")
    for line in list_input_contents(input_dir):
        print(line)

    src: Path | None = None

    if src_override and src_override.exists():
        src = src_override if is_project_root(src_override) else None
        if src:
            print(f"[setup] Usando caminho explícito: {src}")

    if src is None and dataset_slug:
        src = find_kaggle_datasets_path(input_dir, dataset_slug)
        if src:
            print(f"[setup] Encontrado via dataset slug: {src}")

    if src is None:
        src = find_project_root(input_dir)

    if src is None:
        raise FileNotFoundError(
            "\n❌ Projeto Syon NÃO encontrado.\n\n"
            "Seu dataset parece estar em:\n"
            "  /kaggle/input/datasets/<user>/syon-project/Syon/\n\n"
            "Cole na célula 1:\n"
            "  SYON_SRC = Path('/kaggle/input/datasets/regyfelipe/syon-project/Syon')\n"
        )

    print(f"[setup] Projeto: {src}")

    if dest.exists():
        print(f"[setup] Destino já existe: {dest}")
    else:
        shutil.copytree(src, dest)
        print(f"[setup] Copiado → {dest}")

    if not is_project_root(dest):
        raise RuntimeError(f"Cópia incompleta em {dest}")

    return dest


def main() -> None:
    parser = argparse.ArgumentParser(description="Setup Syon no Kaggle")
    parser.add_argument("--input", type=Path, default=Path("/kaggle/input"))
    parser.add_argument("--dest", type=Path, default=Path("/kaggle/working/syon"))
    parser.add_argument("--dataset", type=str, default=None, help="ex: syon-project ou regyfelipe/syon-project")
    parser.add_argument("--src", type=Path, default=None, help="Caminho explícito do Syon/")
    args = parser.parse_args()

    try:
        setup(args.input, args.dest, args.dataset, args.src)
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()