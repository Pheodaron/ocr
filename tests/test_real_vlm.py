"""Opt-in real inference. This smoke test is not an OCR quality benchmark."""

import os

import pytest

from ocr_service.backends import create_engine
from ocr_service.catalog import LABELS
from ocr_service.config import Settings
from ocr_service.processing import process_page

pytestmark = pytest.mark.ocr


@pytest.mark.parametrize("backend", [name for name in LABELS if name != "paddle"])
def test_real_vlm_page(backend, pdf_bytes, tmp_path):
    requested = os.environ.get("OCR_REAL_BACKENDS", "").split(",")
    if os.environ.get("OCR_RUN_MODEL_TESTS") != "1" or backend not in requested:
        pytest.skip("Set OCR_RUN_MODEL_TESTS=1 and OCR_REAL_BACKENDS with prepared model IDs")
    settings = Settings()
    settings.setup()
    source = tmp_path / "source.pdf"
    source.write_bytes(pdf_bytes)
    engine = create_engine(backend, settings)
    try:
        page = process_page(engine, source, 1, tmp_path, settings)
        assert page.text.strip(), "Real model returned no readable text"
        assert "Document" in page.text and "1" in page.text
        assert page.confidence is None
        assert (tmp_path / f"page-1-{page.dpi}.raw.json").is_file()
    finally:
        if hasattr(engine, "close"):
            engine.close()
