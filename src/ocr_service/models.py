"""Explicit model preparation. Inference never resolves remote models."""

import hashlib
import json
import logging
import os
import shutil
import tarfile
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

import yaml

LOGGER = logging.getLogger(__name__)
MODEL_ROLES = {
    "layout": "PP-DocLayout-L",
    "orientation": "PP-LCNet_x1_0_doc_ori",
    "detection": "PP-OCRv5_mobile_det",
    "recognition": "eslav_PP-OCRv5_mobile_rec",
    "table_classification": "PP-LCNet_x1_0_table_cls",
    "wired_structure": "SLANeXt_wired",
    "wireless_structure": "SLANet_plus",
    "wired_cells": "RT-DETR-L_wired_table_cell_det",
    "wireless_cells": "RT-DETR-L_wireless_table_cell_det",
}
MODEL_BASE_URL = (
    "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0"
)


class ModelsUnavailable(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_model(path: Path, name: str) -> None:
    required = ("inference.yml", "inference.pdiparams")
    if not all((path / item).is_file() for item in required) or not any(
        (path / item).is_file() for item in ("inference.json", "inference.pdmodel")
    ):
        raise ModelsUnavailable(f"Модель {name} отсутствует или неполна: {path}")
    config = yaml.safe_load((path / "inference.yml").read_text())
    if config.get("Global", {}).get("model_name") != name:
        raise ModelsUnavailable(f"Неверное имя модели в {path / 'inference.yml'}")


def verify_models(directory: Path) -> dict:
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise ModelsUnavailable("Нет manifest.json. Выполните prepare-models перед запуском worker")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("format_version") != 1:
        raise ModelsUnavailable("Неизвестная версия manifest.json")
    for name in MODEL_ROLES.values():
        path = directory / name
        validate_model(path, name)
        hashes = manifest.get("models", {}).get(name, {}).get("files", {})
        required = {"inference.yml", "inference.pdiparams"}
        required.add(
            "inference.json" if (path / "inference.json").is_file() else "inference.pdmodel"
        )
        if not required.issubset(hashes):
            raise ModelsUnavailable(f"Нет контрольных сумм модели {name}")
        for filename, expected in hashes.items():
            relative = Path(filename)
            if relative.is_absolute() or ".." in relative.parts:
                raise ModelsUnavailable("Недопустимый путь в manifest.json")
            target = path / relative
            if not target.is_file() or sha256_file(target) != expected:
                raise ModelsUnavailable(f"Не совпала SHA256: {name}/{filename}")
    return manifest


def prepare_models(directory: Path, allow_download: bool = False) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "manifest.json").is_file():
        return verify_models(directory)
    records = {}
    for name in MODEL_ROLES.values():
        destination = directory / name
        url = f"{MODEL_BASE_URL}/{name}_infer.tar"
        archive_digest = None
        if not destination.exists():
            if not allow_download:
                raise ModelsUnavailable(
                    f"Нет {name}. Подготовьте модели с сетью: prepare-models --download"
                )
            LOGGER.info("Downloading %s", name)
            with tempfile.TemporaryDirectory(prefix=".prepare-", dir=directory) as temporary:
                temporary = Path(temporary)
                archive_path = temporary / "model.tar"
                try:
                    with urllib.request.urlopen(url, timeout=120) as response:
                        with archive_path.open("wb") as output:
                            total = 0
                            for chunk in iter(lambda: response.read(1024 * 1024), b""):
                                total += len(chunk)
                                if total > 2 * 1024**3:
                                    raise ModelsUnavailable(f"Слишком большой архив {name}")
                                output.write(chunk)
                except (urllib.error.URLError, TimeoutError) as error:
                    raise ModelsUnavailable(
                        f"Не удалось скачать {name} с {url}: {error}. "
                        "Проверьте доступ к серверу или импортируйте готовый каталог моделей."
                    ) from error
                archive_digest = sha256_file(archive_path)
                extracted = temporary / "unpacked"
                extracted.mkdir()
                with tarfile.open(archive_path) as archive:
                    members = archive.getmembers()
                    if sum(item.size for item in members) > 4 * 1024**3:
                        raise ModelsUnavailable(f"Слишком большой распакованный архив {name}")
                    if any(not (item.isfile() or item.isdir()) for item in members):
                        raise ModelsUnavailable(f"Недопустимые ссылки в архиве {name}")
                    archive.extractall(extracted, filter="data")
                configs = list(extracted.rglob("inference.yml"))
                if len(configs) != 1:
                    raise ModelsUnavailable(f"Неоднозначное содержимое архива {name}")
                source = configs[0].parent
                validate_model(source, name)
                source.rename(destination)
        validate_model(destination, name)
        records[name] = {
            "source": url if archive_digest else "pre-existing local directory",
            "archive_sha256": archive_digest,
            "files": {
                str(path.relative_to(destination)): sha256_file(path)
                for path in sorted(destination.rglob("*"))
                if path.is_file()
            },
        }
    manifest = {"format_version": 1, "models": records}
    temporary_manifest = directory / "manifest.json.tmp"
    temporary_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    os.replace(temporary_manifest, directory / "manifest.json")
    return manifest


def configure_offline_environment(cpu_threads: int, cache_dir: Path) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["PADDLE_PDX_CACHE_HOME"] = str(cache_dir.resolve())
    os.environ["OMP_NUM_THREADS"] = str(cpu_threads)
    os.environ["MKL_NUM_THREADS"] = str(cpu_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(cpu_threads)


def copy_local_models(source: Path, destination: Path) -> None:
    """Import an already prepared model directory into a Compose volume."""
    verify_models(source)
    if (destination / "manifest.json").is_file():
        verify_models(destination)
        return
    destination.mkdir(parents=True, exist_ok=True)
    for name in MODEL_ROLES.values():
        shutil.copytree(source / name, destination / name, dirs_exist_ok=True)
    shutil.copy2(source / "manifest.json", destination / "manifest.json")
    verify_models(destination)
