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
DOTS_REVISION = "fc5b04fbc134d7ee74dfd65b46193f7056db3e1b"
GLM_REPO = "ggml-org/GLM-OCR-GGUF"
GLM_REVISION = "65a42de1148dbed2297e922b5dbc7d9b70c36578"


def artifacts(backend: str, settings: Settings) -> dict:
    additional = {
        "lighton_small": (
            "prithivMLmods/LightOnOCR-3-0.8B-GGUF",
            "414757bdb78fd07c947a1e83c2b8d7e9a8b73583",
            "Q8_0",
            "LightOnOCR-3-0.8B.Q8_0.gguf",
            "LightOnOCR-3-0.8B.mmproj-q8_0.gguf",
            "660a8931ba33c8a604076f69b8785b36eadcdb1cc90004e9f89bf67615a0456d",
            "a8b9b3b01b6c780747b48e0807ef5b753c54f27aa4c23962c3c7d5ace2ed0233",
        ),
        "jina": (
            "ahmadw/jina-ocr-v1-GGUF",
            "333253b2ff8f3d9357d155327bee5dc98e007829",
            "Q8_0",
            "jina-ocr-v1-Q8_0.gguf",
            "mmproj-jina-ocr-v1-F16.gguf",
            "2612fc330ad777ffbbd2a095aad41081ad18be70bfe1bcf42b3a295fad0b6041",
            "3a2bdc5303c9845e5d35e5175781d7eb81bf28ee212170944cf3ad9d27d37c01",
        ),
        "surya": (
            "datalab-to/surya-ocr-2-gguf",
            "6a3a4c30e5e74446d4f8b6afd05b2f2da970f470",
            "F16",
            "surya-2.gguf",
            "surya-2-mmproj.gguf",
            "1f18abe17b1ed8b4e47ee9b1ad0e274c93daf5efbb6b29a04ff1712e37051e05",
            "98c0563673b1657ff6d021d1e5f04af06cbf61bb40c63ac613e8bb71b42fb2c0",
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
    if backend in {"chandra", "lighton"}:
        if backend == "chandra":
            repo, revision, quant, name = (
                "prithivMLmods/chandra-ocr-2-GGUF",
                "85cc7b75893f81669d6751ba6d31c718c7c98e1d",
                "Q5_K_M",
                "chandra-ocr-2",
            )
            hashes = (
                "c5e3df32bfa20d0a822fdc635d3849f5d53e2049535cd85753df4fd762bdd276",
                "709be8aeda77c5732e58e957367439eae14ebb73b4f4a78b558013d000948bc3",
            )
        else:
            repo, revision, quant, name = (
                "prithivMLmods/LightOnOCR-3-1B-GGUF",
                "ff5c331ec2cfee9de1c8054c21ccf88b8c5b0faa",
                "Q8_0",
                "LightOnOCR-3-1B",
            )
            hashes = (
                "2cf9771aaf236a1c25f389a344510c9ad4daa863e5fdd82651cf7d7834fede1c",
                "7bbf40e6f3e91626843fdcbba67c7fe774821f3f0c537dcc8c769026a528839a",
            )
        model, projector = f"{name}.{quant}.gguf", f"{name}.mmproj-q8_0.gguf"
        return {
            "repo": repo,
            "revision": revision,
            "quantization": quant,
            "model": model,
            "projector": projector,
            "files": dict(zip((model, projector), hashes, strict=True)),
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
            "revision": DOTS_REVISION,
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
            except (OSError, urllib.error.URLError) as error:
                raise ModelsUnavailable(f"Не удалось подготовить {filename}: {error}") from error
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
