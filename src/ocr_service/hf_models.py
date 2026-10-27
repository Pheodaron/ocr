"""Pinned local snapshots for models shipping their own Transformers implementation."""

import json
import os
import shutil
from pathlib import Path

from .config import Settings
from .models import ModelsUnavailable, sha256_file

SPECS = {
    "falcon": ("tiiuae/Falcon-OCR", "fe757d59ecd79d4d68760162306a70a015761ad9"),
}


def prepare_hf(backend: str, settings: Settings, download=False, source: Path | None = None):
    repo, revision = SPECS[backend]
    directory = settings.models_dir / backend
    if (directory / "manifest.json").is_file():
        return verify_hf(backend, settings)
    directory.mkdir(parents=True, exist_ok=True)
    try:
        if source:
            shutil.copytree(source / backend, directory, dirs_exist_ok=True)
            return verify_hf(backend, settings)
        if download:
            # Preparation only. Runtime always passes local_files_only=True and is offline.
            os.environ["HF_HUB_OFFLINE"] = "0"
            from huggingface_hub import snapshot_download

            snapshot_download(
                repo_id=repo,
                revision=revision,
                local_dir=directory,
                allow_patterns=[
                    "*.json",
                    "*.jinja",
                    "*.py",
                    "*.safetensors",
                    "*.txt",
                    "*.md",
                    "*.model",
                    "LICENSE*",
                    "*ATTRIBUTION*",
                    "*NOTICE*",
                ],
                max_workers=2,
            )
        required = ["config.json", "model.safetensors", "tokenizer_config.json"]
        if not all((directory / name).is_file() for name in required):
            raise ModelsUnavailable(f"Нет локального snapshot {backend}; prepare-models --download")
        records = {
            str(path.relative_to(directory)): sha256_file(path)
            for path in directory.rglob("*")
            if path.is_file()
            and ".cache" not in path.relative_to(directory).parts
            and path.name not in {"manifest.json", "manifest.json.tmp"}
        }
        manifest = {"format_version": 1, "repo": repo, "revision": revision, "files": records}
        temporary = directory / "manifest.json.tmp"
        temporary.write_text(json.dumps(manifest, indent=2))
        os.replace(temporary, directory / "manifest.json")
        return verify_hf(backend, settings)
    except ModelsUnavailable:
        raise
    except Exception as error:
        raise ModelsUnavailable(f"Не удалось подготовить {backend}: {error}") from error


def verify_hf(backend: str, settings: Settings) -> dict:
    repo, revision = SPECS[backend]
    directory = settings.models_dir / backend
    try:
        manifest = json.loads((directory / "manifest.json").read_text())
        if manifest.get("revision") != revision or manifest.get("repo") != repo:
            raise ModelsUnavailable(f"Неверный snapshot {backend}")
        if not {"config.json", "model.safetensors", "tokenizer_config.json"}.issubset(
            manifest["files"]
        ):
            raise ModelsUnavailable(f"Неполный manifest {backend}")
        for filename, expected in manifest["files"].items():
            path = Path(filename)
            if path.is_absolute() or ".." in path.parts:
                raise ModelsUnavailable("Недопустимый путь manifest")
            if sha256_file(directory / path) != expected:
                raise ModelsUnavailable(f"Не совпала SHA256: {backend}/{filename}")
    except (OSError, ValueError, KeyError) as error:
        raise ModelsUnavailable(f"Snapshot {backend} не подготовлен: {error}") from error
    return manifest
