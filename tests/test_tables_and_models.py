import json

import pytest
import yaml

from ocr_service.engine import pipeline_config
from ocr_service.models import MODEL_ROLES, ModelsUnavailable, prepare_models, verify_models
from ocr_service.tables import parse_table


def test_html_table_spans_and_identifier_preservation():
    markup = """<table onclick="bad()"><tr><th rowspan="2">00125</th>
        <th colspan="2">Заголовок</th></tr><tr><td>A</td><td>B</td></tr>
        <tr><td>0</td><td></td><td>1,25</td></tr></table>"""
    safe, cells = parse_table(markup)
    assert "onclick" not in safe
    assert cells[0].text == "00125"
    assert cells[0].rowspan == 2
    assert [(cell.row, cell.column) for cell in cells[:4]] == [(0, 0), (0, 1), (1, 1), (1, 2)]
    assert cells[-2].text == ""
    assert cells[-1].text == "1,25"


def test_table_sanitizer_removes_executable_elements_and_attributes():
    safe, cells = parse_table(
        '<table><tr><td><script>alert(1)</script><img src="https://external/image">'
        '<a href="javascript:bad()">текст</a></td></tr></table>'
    )
    assert "script" not in safe and "img" not in safe and "href" not in safe
    assert cells[0].text == "текст"


def test_table_line_breaks_and_invalid_markup():
    _, cells = parse_table("<table><tr><td>Максимальная<br>мощность</td></tr></table>")
    assert cells[0].text == "Максимальная мощность"
    for markup in ("", "<table></table>", "<table><tr><td colspan='huge'>x</td></tr></table>"):
        with pytest.raises(ValueError):
            parse_table(markup)


def test_pipeline_config_is_offline_including_lazy_backup(settings):
    config = pipeline_config(settings)
    assert not config["use_formula_recognition"]
    assert not config["use_region_detection"]
    assert not config["use_chart_recognition"]
    assert config["markdown_ignore_labels"] == []
    pipelines = config["SubPipelines"]
    primary = pipelines["GeneralOCR"]
    backup = pipelines["TableRecognition"]["SubPipelines"]["GeneralOCR"]
    for ocr in (primary, backup):
        assert not ocr["use_textline_orientation"]
        model = ocr["SubModules"]["TextRecognition"]
        assert model["model_name"] == "eslav_PP-OCRv5_mobile_rec"
        assert model["model_dir"].startswith(str(settings.models_dir))
        assert model["score_thresh"] == 0
    assert config["SubModules"]["LayoutDetection"]["layout_merge_bboxes_mode"] == "union"


def test_models_require_explicit_preparation_and_detect_changed_weights(settings):
    directory = settings.models_dir
    with pytest.raises(ModelsUnavailable):
        verify_models(directory)
    with pytest.raises(ModelsUnavailable):
        prepare_models(directory)
    for name in MODEL_ROLES.values():
        path = directory / name
        path.mkdir()
        (path / "inference.yml").write_text(yaml.safe_dump({"Global": {"model_name": name}}))
        (path / "inference.json").write_text("{}")
        (path / "inference.pdiparams").write_bytes(b"test weights, not a real OCR model")
    manifest = prepare_models(directory)
    assert verify_models(directory) == manifest
    assert len(manifest["models"]) == 9
    name = MODEL_ROLES["recognition"]
    (directory / name / "inference.pdiparams").write_bytes(b"changed")
    with pytest.raises(ModelsUnavailable, match="SHA256"):
        verify_models(directory)
    saved = json.loads((directory / "manifest.json").read_text())
    assert saved["models"][name]["source"] == "pre-existing local directory"
