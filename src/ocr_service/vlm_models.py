"""Pinned GGUF artifacts, including each model's own vision encoder/projector."""

import json
import logging
import os
import shutil
import urllib.error
import urllib.request
from pathlib import Path

from .config import Settings
from .models import ModelsUnavailable, sha256_file

LOGGER = logging.getLogger(__name__)
DOTS_REPO = "lodrick-the-lafted/dots.mocr-gguf"
# Q5_K_M and BF16 were added after the original Q8-only upload.
DOTS_REVISION = "8bc6c189780084cf59b10d55fadcbe7d1292696a"
DOTS_Q8_REVISION = "fc5b04fbc134d7ee74dfd65b46193f7056db3e1b"
GLM_REPO = "ggml-org/GLM-OCR-GGUF"
GLM_REVISION = "65a42de1148dbed2297e922b5dbc7d9b70c36578"


def artifacts(backend: str, settings: Settings) -> dict:
    additional = {
        "lighton_small": (
            "prithivMLmods/LightOnOCR-3-0.8B-GGUF",
            "414757bdb78fd07c947a1e83c2b8d7e9a8b73583",
            settings.lighton_small_quantization,
            f"LightOnOCR-3-0.8B.{settings.lighton_small_quantization}.gguf",
            "LightOnOCR-3-0.8B.mmproj-q8_0.gguf",
            {
                "Q8_0": "660a8931ba33c8a604076f69b8785b36eadcdb1cc90004e9f89bf67615a0456d",
                "Q4_K_M": "290ba9b31716a00dc1cbe1555a2776109f21ae3f4f5c177a2b543c9bc3eab12a",
            }[settings.lighton_small_quantization],
            "a8b9b3b01b6c780747b48e0807ef5b753c54f27aa4c23962c3c7d5ace2ed0233",
        ),
    }
    if backend in additional:
        repo, revision, quant, model, projector, model_hash, projector_hash = additional[backend]
        return {
            "repo": repo,
            "revision": revision,
            "quantization": quant,
            "model": model,
            "projector": projector,
            "files": {model: model_hash, projector: projector_hash},
        }
    if backend == "dots_mocr":
        quant = settings.dots_quantization
        filename, digest = {
            "Q5_K_M": (
                "dotsmocr-1.8b-q5_k_m.gguf",
                "ce7138e2e32ee2b930181150d7c8f58d1e9a1c785ee47873521a6e1a573a83c6",
            ),
            "Q8_0": (
                "dotsmocr-1.8b-q8_0.gguf",
                "779cff469ce177d3e0f6455408b25755bd40a23aaa72b6d5ad7a48b9a05deb46",
            ),
            # BF16 integrity is recorded on preparation, as with Paddle weights.
            "BF16": ("dotsmocr-1.8b-bf16.gguf", None),
        }[quant]
        return {
            "repo": DOTS_REPO,
            "revision": DOTS_Q8_REVISION if quant == "Q8_0" else DOTS_REVISION,
            "quantization": quant,
            "model": filename,
            "projector": "mmproj-dotsmocr-q8_0.gguf",
            "files": {
                filename: digest,
                "mmproj-dotsmocr-q8_0.gguf": (
                    "257f11275143d3d4d0afa5336ff409a9550bc3a19cf3252e7beee0ab39a9b042"
                ),
            },
        }
    if backend != "glm_ocr":
        raise ValueError(backend)
    return {
        "repo": GLM_REPO,
        "revision": GLM_REVISION,
        "quantization": "Q8_0",
        "model": "GLM-OCR-Q8_0.gguf",
        "projector": "mmproj-GLM-OCR-Q8_0.gguf",
        "files": {
            "GLM-OCR-Q8_0.gguf": "45bc244a6446aff850521dc41f18bc8d7105ad5f0c2c8c28af04e7cc4f4d50b1",
            "mmproj-GLM-OCR-Q8_0.gguf": (
                "9c4b58e33e316ed142eb5dcb41abec3844d3e6e5dc361ffb782c3fa9d175141f"
            ),
        },
    }


def model_directory(backend: str, settings: Settings) -> Path:
    return settings.models_dir / backend


def prepare_vlm(
    backend: str, settings: Settings, *, download: bool = False, source: Path | None = None
) -> dict:
    spec = artifacts(backend, settings)
    directory = model_directory(backend, settings)
    directory.mkdir(parents=True, exist_ok=True)
    records = {}
    for filename, pinned_hash in spec["files"].items():
        target = directory / filename
        url = f"https://huggingface.co/{spec['repo']}/resolve/{spec['revision']}/{filename}"
        if not target.exists():
            temporary = target.with_suffix(".part")
            try:
                if source:
                    shutil.copyfile(source / backend / filename, temporary)
                elif download:
                    LOGGER.info("Downloading %s/%s", backend, filename)
                    with urllib.request.urlopen(url, timeout=120) as response:
                        with temporary.open("wb") as output:
                            total = 0
                            for chunk in iter(lambda: response.read(1024 * 1024), b""):
                                total += len(chunk)
                                if total > 6 * 1024**3:
                                    raise ModelsUnavailable(f"Слишком большой GGUF: {filename}")
                                output.write(chunk)
                else:
                    raise ModelsUnavailable(f"Нет {target}. Выполните prepare-models --download")
                _verify_file(temporary, pinned_hash)
                os.replace(temporary, target)
            except urllib.error.HTTPError as error:
                hint = "Файл отсутствует в выбранной ревизии модели. " if error.code == 404 else ""
                raise ModelsUnavailable(
                    f"Не удалось подготовить {filename}: HTTP {error.code}. {hint}URL: {url}"
                ) from error
            except (OSError, urllib.error.URLError) as error:
                raise ModelsUnavailable(
                    f"Не удалось подготовить {filename} ({url}): {error}"
                ) from error
            finally:
                temporary.unlink(missing_ok=True)
        records[filename] = _verify_file(target, pinned_hash)
    manifest = {**spec, "format_version": 1, "files": records}
    path = directory / f"manifest-{spec['quantization']}.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    os.replace(temporary, path)
    return manifest


def _verify_file(path: Path, expected: str | None) -> str:
    with path.open("rb") as stream:
        if stream.read(4) != b"GGUF":
            raise ModelsUnavailable(f"Не GGUF: {path}")
    digest = sha256_file(path)
    if expected and expected != digest:
        raise ModelsUnavailable(f"Не совпала SHA256: {path.name}")
    return digest


def verify_vlm(backend: str, settings: Settings) -> dict:
    spec = artifacts(backend, settings)
    directory = model_directory(backend, settings)
    path = directory / f"manifest-{spec['quantization']}.json"
    try:
        manifest = json.loads(path.read_text())
        if manifest.get("format_version") != 1 or any(
            manifest.get(key) != spec[key] for key in ("repo", "revision", "model", "projector")
        ):
            raise ModelsUnavailable(f"Неверный manifest: {path}")
        for filename, pinned_hash in spec["files"].items():
            expected = manifest["files"][filename]
            if pinned_hash and expected != pinned_hash:
                raise ModelsUnavailable(
                    f"Manifest не соответствует закреплённой SHA256: {filename}"
                )
            _verify_file(directory / filename, expected)
    except (OSError, ValueError, KeyError) as error:
        raise ModelsUnavailable(f"Модели {backend} не подготовлены: {error}") from error
    return manifest
