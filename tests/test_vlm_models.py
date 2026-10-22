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


@pytest.mark.parametrize("quantization", ["Q5_K_M", "Q8_0", "BF16"])
def test_dots_artifacts_exist_in_the_selected_upstream_snapshot(settings, quantization):
    from ocr_service.vlm_models import artifacts

    # Upstream inventories checked against HF tree/commit pages, 2026-10-22.
    # fc5b04f predates both Q5_K_M and BF16: pinning every quant to it caused HTTP 404.
    original = {"dotsmocr-1.8b-q8_0.gguf", "mmproj-dotsmocr-q8_0.gguf"}
    published = {
        "fc5b04fbc134d7ee74dfd65b46193f7056db3e1b": original,
        "8bc6c189780084cf59b10d55fadcbe7d1292696a": original
        | {
            "dotsmocr-1.8b-q5_k_m.gguf",
            "dotsmocr-1.8b-bf16.gguf",
            "mmproj-dotsmocr-bf16.gguf",
        },
    }
    spec = artifacts("dots_mocr", settings.model_copy(update={"dots_quantization": quantization}))
    assert set(spec["files"]) <= published[spec["revision"]]
    assert spec["projector"] == "mmproj-dotsmocr-q8_0.gguf"


def test_failed_download_reports_exact_url_and_cleans_partial_file(settings, monkeypatch):
    import urllib.error

    from ocr_service import vlm_models

    spec = vlm_models.artifacts("dots_mocr", settings)
    expected = f"https://huggingface.co/{spec['repo']}/resolve/{spec['revision']}/{spec['model']}"

    def missing_file(url, **kwargs):
        assert url == expected
        raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)

    monkeypatch.setattr(vlm_models.urllib.request, "urlopen", missing_file)
    with pytest.raises(ModelsUnavailable, match="Файл отсутствует") as error:
        prepare_vlm("dots_mocr", settings, download=True)
    assert expected in str(error.value)
    assert not list((settings.models_dir / "dots_mocr").glob("*.part"))
    assert not list((settings.models_dir / "dots_mocr").glob("manifest*"))
