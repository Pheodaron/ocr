import hashlib
import json

import pytest

from ocr_service.models import ModelsUnavailable
from ocr_service.vlm_models import prepare_vlm, verify_vlm


def test_gguf_preparation_requires_matching_projector_and_checks_integrity(settings, monkeypatch):
    from ocr_service import vlm_models

    model = b"GGUFmodel-fixture"
    projector = b"GGUFprojector-fixture"
    spec = {
        "repo": "test/fixture",
        "revision": "fixture",
        "quantization": "Q8_0",
        "model": "model.gguf",
        "projector": "mmproj.gguf",
        "files": {
            "model.gguf": hashlib.sha256(model).hexdigest(),
            "mmproj.gguf": hashlib.sha256(projector).hexdigest(),
        },
    }
    monkeypatch.setattr(vlm_models, "artifacts", lambda *_args: spec)
    directory = settings.models_dir / "glm_ocr"
    directory.mkdir(parents=True)
    (directory / "model.gguf").write_bytes(model)
    with pytest.raises(ModelsUnavailable, match="prepare-models"):
        prepare_vlm("glm_ocr", settings)
    (directory / "mmproj.gguf").write_bytes(projector)
    manifest = prepare_vlm("glm_ocr", settings)
    assert verify_vlm("glm_ocr", settings) == manifest
    (directory / "mmproj.gguf").write_bytes(b"GGUFaltered")
    with pytest.raises(ModelsUnavailable, match="SHA256"):
        verify_vlm("glm_ocr", settings)
    # Modifying the manifest alongside a file cannot defeat a pinned upstream hash.
    manifest["files"]["mmproj.gguf"] = hashlib.sha256(b"GGUFaltered").hexdigest()
    (directory / "manifest-Q8_0.json").write_text(json.dumps(manifest))
    with pytest.raises(ModelsUnavailable, match="SHA256"):
        verify_vlm("glm_ocr", settings)
