import os
from pathlib import Path

import pymupdf
import pytest

from ocr_service.config import Settings
from ocr_service.engine import PaddleEngine
from ocr_service.processing import process_page


@pytest.mark.ocr
@pytest.mark.skipif(os.getenv("OCR_RUN_MODEL_TESTS") != "1", reason="Set OCR_RUN_MODEL_TESTS=1")
def test_real_russian_pdf_and_table(tmp_path):
    settings = Settings()
    font = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    if not font.is_file():
        pytest.skip("DejaVuSans.ttf is needed for the generated Russian test PDF")
    source = tmp_path / "sample.pdf"
    with pymupdf.open() as document:
        page = document.new_page()
        page.insert_font(fontname="dejavu", fontfile=str(font))
        page.insert_text((50, 70), "Договор 00125", fontname="dejavu", fontsize=20)
        for y in (130, 180, 230):
            page.draw_line((50, y), (530, y), color=(0, 0, 0))
        for x in (50, 290, 530):
            page.draw_line((x, 130), (x, 230), color=(0, 0, 0))
        for x, y, text in [
            (65, 160, "Показатель"),
            (305, 160, "Значение"),
            (65, 210, "Мощность"),
            (305, 210, "150 кВт"),
        ]:
            page.insert_text((x, y), text, fontname="dejavu", fontsize=16)
        document.save(source)
    result = process_page(PaddleEngine(settings), source, 1, tmp_path, settings)
    assert "00125" in result.text
    assert "договор" in result.text.lower()
    assert result.tables, "The bordered table must be detected and reconstructed"
    assert any("150" in cell.text for table in result.tables for cell in table.cells)
