"""Явная подготовка моделей при наличии интернета; рабочие контейнеры их не скачивают."""

import argparse
import hashlib
import json
from pathlib import Path

MODELS = {
    "lighton4b": ("lightonai/LightOnOCR-3-4B", None),
    "lighton08": ("lightonai/LightOnOCR-3-0.8B", None),
    "qwen": ("unsloth/Qwen3.8-27B-GGUF", ["Qwen3.8-27B-UD-Q3_K_XL.gguf", "mmproj-F16.gguf"]),
}


def checksum(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=[*MODELS, "all"])
    parser.add_argument("--root", type=Path, default=Path("models"))
    parser.add_argument(
        "--revision", help="Ревизия одной выбранной модели; без неё main разрешается в точную SHA"
    )
    parser.add_argument("--verify", action="store_true", help="Только локальная проверка без сети")
    args = parser.parse_args()
    if args.model == "all" and args.revision:
        parser.error("--revision задаётся отдельно для каждой модели")
    for name in MODELS if args.model == "all" else [args.model]:
        directory = args.root / name
        manifest_path = directory / "download-manifest.json"
        if args.verify:
            manifest = json.loads(manifest_path.read_text())
            for relative, expected in manifest["files"].items():
                if checksum(directory / relative) != expected:
                    raise RuntimeError(f"Повреждён файл {name}/{relative}")
            print(f"{name}: контрольные суммы подтверждены")
            continue
        from huggingface_hub import HfApi, snapshot_download

        repository, patterns = MODELS[name]
        revision = HfApi().model_info(repository, revision=args.revision).sha
        print(f"Подготовка {repository}@{revision}")
        snapshot_download(
            repository,
            revision=revision,
            local_dir=directory,
            allow_patterns=patterns,
            ignore_patterns=["*.md", ".gitattributes"],
        )
        files = {
            str(path.relative_to(directory)): checksum(path)
            for path in directory.rglob("*")
            if path.is_file() and ".cache" not in path.parts and path != manifest_path
        }
        manifest_path.write_text(
            json.dumps({"repository": repository, "revision": revision, "files": files}, indent=2)
        )
        print(f"{name}: сохранено {len(files)} файлов")


if __name__ == "__main__":
    main()
