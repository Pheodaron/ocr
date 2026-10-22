import pymupdf
import pytest
from fastapi.testclient import TestClient

from ocr_service.api import create_app
from ocr_service.config import Settings


@pytest.fixture
def settings(tmp_path):
    return Settings(
        data_dir=tmp_path / "data",
        models_dir=tmp_path / "models",
        dpi=100,
        retry_dpi=140,
        max_upload_mb=1,
        max_pages=5,
    )


@pytest.fixture
def client(settings):
    with TestClient(create_app(settings)) as client:
        yield client


@pytest.fixture
def pdf_bytes():
    document = pymupdf.open()
    for number in range(1, 4):
        page = document.new_page(width=300, height=300)
        page.insert_text((30, 50), f"Document page {number}", fontsize=16)
    content = document.tobytes()
    document.close()
    return content


@pytest.fixture
def prediction():
    return {
        "res": {
            "doc_preprocessor_res": {"angle": 0},
            "overall_ocr_res": {
                "rec_texts": ["Договор 00125", "Мощность", "150 кВт"],
                "rec_scores": [0.98, 0.97, 0.96],
                "rec_boxes": [[10, 10, 250, 40], [10, 60, 100, 90], [100, 60, 250, 90]],
            },
            "parsing_res_list": [
                {
                    "block_label": "doc_title",
                    "block_content": "Договор 00125",
                    "block_bbox": [10, 10, 250, 40],
                    "block_order": 0,
                },
                {
                    "block_label": "table",
                    "block_content": "<table><tr><th>Мощность</th><td>150 кВт</td></tr></table>",
                    "block_bbox": [10, 60, 250, 90],
                    "block_order": 1,
                },
                {
                    "block_label": "footer",
                    "block_content": "Примечание",
                    "block_bbox": [10, 280, 250, 295],
                    "block_order": None,
                },
            ],
            "table_res_list": [
                {
                    "pred_html": "<table><tr><th>Мощность</th><td>150 кВт</td></tr></table>",
                    "cell_box_list": [],
                    "table_ocr_pred": {},
                }
            ],
        }
    }
