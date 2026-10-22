"""Explicit preparation of the original KDL checkpoint for portable CPU inference."""

import json
import os
import shutil
import urllib.request
from pathlib import Path

from .config import Settings
from .models import ModelsUnavailable, sha256_file

REPO = "KDLAI/KDL-Frontier-Parser-nano"
REVISION = "a6cb7d2a9ac5fd13f527764f5411c9ec3ad2c4ec"
FILES = (
    "config.json",
    "model.safetensors",
    "generation_config.json",
    "processor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "README.md",
)


def prepare_kdl(settings: Settings, download: bool = False, source: Path | None = None) -> dict:
    directory = settings.models_dir / "kdl"
    if (directory / "manifest.json").is_file():
        return verify_kdl(settings)
    directory.mkdir(parents=True, exist_ok=True)
    records = {}
    for filename in FILES:
        target = directory / filename
        temporary = target.with_suffix(".part")
        if not target.is_file():
            try:
                if source:
                    shutil.copyfile(source / "kdl" / filename, temporary)
                elif download:
                    url = f"https://huggingface.co/{REPO}/resolve/{REVISION}/{filename}"
                    with urllib.request.urlopen(url, timeout=120) as response:
                        with temporary.open("wb") as output:
                            total = 0
                            for chunk in iter(lambda: response.read(1024 * 1024), b""):
                                total += len(chunk)
                                if total > 4 * 1024**3:
                                    raise ModelsUnavailable(f"Слишком большой файл KDL: {filename}")
                                output.write(chunk)
                else:
                    raise ModelsUnavailable("Нет KDL. Выполните prepare-models --download")
                os.replace(temporary, target)
            except OSError as error:
                raise ModelsUnavailable(
                    f"Не удалось подготовить KDL {filename}: {error}"
                ) from error
            finally:
                temporary.unlink(missing_ok=True)
        records[filename] = sha256_file(target)
    manifest = {"format_version": 1, "repo": REPO, "revision": REVISION, "files": records}
    path = directory / "manifest.json.tmp"
    path.write_text(json.dumps(manifest, indent=2))
    os.replace(path, directory / "manifest.json")
    return verify_kdl(settings)


def verify_kdl(settings: Settings) -> dict:
    directory = settings.models_dir / "kdl"
    try:
        manifest = json.loads((directory / "manifest.json").read_text())
        if manifest.get("format_version") != 1 or manifest.get("revision") != REVISION:
            raise ModelsUnavailable("Неверный manifest KDL")
        for filename in FILES:
            if sha256_file(directory / filename) != manifest["files"][filename]:
                raise ModelsUnavailable(f"Не совпала SHA256 KDL {filename}")
        config = json.loads((directory / "config.json").read_text())
        if config.get("model_type") != "qwen2_vl":
            raise ModelsUnavailable("Ожидалась архитектура KDL qwen2_vl")
    except (OSError, KeyError, ValueError) as error:
        raise ModelsUnavailable(f"KDL не подготовлена: {error}") from error
    return manifest
