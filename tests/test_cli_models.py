import sys
from pathlib import Path

import pytest

from ocr_service import cli
from ocr_service.catalog import LABELS


def test_prepare_selected_models_uses_shared_directory_and_env_quantization(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text(
        "OCR_MODELS_DIR=shared-models\nOCR_DOTS_QUANTIZATION=Q8_0\nOCR_ENABLED_BACKENDS=paddle\n"
    )
    monkeypatch.setattr(sys, "argv", ["ocr-service", "prepare-models", "--download", "dots_mocr"])
    calls = []

    def prepare(backend, settings, **kwargs):
        calls.append((backend, settings.models_dir, settings.dots_quantization, kwargs))
        return {}

    monkeypatch.setattr("ocr_service.vlm_models.prepare_vlm", prepare)
    monkeypatch.setattr(
        "ocr_service.models.prepare_models",
        lambda *args, **kwargs: pytest.fail("Unselected Paddle must not be downloaded"),
    )
    cli.main()
    assert calls == [
        (
            "dots_mocr",
            Path("shared-models"),
            "Q8_0",
            {"download": True, "source": None},
        )
    ]


def test_prepare_default_includes_all_current_models(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OCR_ENABLED_BACKENDS", "paddle")
    monkeypatch.setattr(sys, "argv", ["ocr-service", "prepare-models", "--download"])
    calls = []
    monkeypatch.setattr(
        "ocr_service.models.prepare_models",
        lambda *args, **kwargs: calls.append("paddle") or {},
    )
    monkeypatch.setattr(
        "ocr_service.vlm_models.prepare_vlm",
        lambda backend, *args, **kwargs: calls.append(backend) or {},
    )
    cli.main()
    assert calls == list(LABELS)


def test_prepare_unknown_backend_fails_before_loading(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["ocr-service", "prepare-models", "jina"])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2
