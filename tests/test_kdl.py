import numpy as np
import pytest

from ocr_service.kdl import KDLEngine, layout_regions
from ocr_service.otsl import otsl_to_html
from ocr_service.tables import parse_table
from ocr_service.vlm_adapter import adapt_kdl


def test_otsl_preserves_rectangular_merges_empty_cells_and_zeros():
    markup = otsl_to_html("<otsl><fcel>00125<lcel><fcel>150 кВт<nl><ucel><xcel><ecel><nl></otsl>")
    _safe, cells = parse_table(markup)
    assert cells[0].text == "00125"
    assert (cells[0].rowspan, cells[0].colspan) == (2, 2)
    assert cells[1].text == "150 кВт"
    assert cells[2].text == "" and cells[2].column == 2


@pytest.mark.parametrize(
    "content",
    [
        "<ucel><nl>",
        "<fcel>A<nl><xcel><nl>",
        "<fcel>A<lcel><nl><ucel><fcel>B<nl>",
        "<otsl><fcel><otsl><fcel>nested<nl></otsl><nl></otsl>",
    ],
)
def test_malformed_otsl_does_not_invent_structure(content):
    with pytest.raises(ValueError):
        otsl_to_html(content)


def test_kdl_uses_layout_then_task_specific_crops_and_survives_bad_region(settings):
    calls = []
    engine = KDLEngine.__new__(KDLEngine)
    engine.settings = settings

    def generate(image, prompt, _budget):
        calls.append((image.size, prompt))
        if "Layout" in prompt:
            text = (
                "<|box_start|>0 0 1000 200<|box_end|><|ref_start|>text<|ref_end|>"
                "<|box_start|>0 200 1000 400<|box_end|><|ref_start|>table<|ref_end|>"
                "<|box_start|>0 400 1000 1000<|box_end|><|ref_start|>table<|ref_end|>"
            )
        elif len(calls) == 3:
            raise ValueError("bad region fixture")
        elif "Table" in prompt:
            text = "<fcel>00125<fcel>150 кВт<nl>"
        else:
            text = "Договор"
        return {
            "text": text,
            "completion_tokens": 5,
            "prompt_tokens": 10,
            "seconds": 0.5,
            "truncated": False,
        }

    engine.generate = generate
    prediction = engine.recognize(np.zeros((200, 100, 3), dtype=np.uint8))
    assert calls[0] == ((1036, 1036), "\nLayout Detection:\n")
    assert calls[1] == ((100, 40), "\nText Recognition:\n")
    assert calls[3] == ((100, 120), "\nTable Recognition:\n")
    page = adapt_kdl(prediction.raw, 1, 250, 100, 200)
    assert page.tables[0].cells[0].text == "00125"
    assert page.tables[0].bbox == [0, 80, 100, 200]
    assert any("bad region fixture" in warning for warning in page.warnings)
    assert page.metrics["calls"] == 3


def test_kdl_layout_limits_and_bounds():
    text = "<|box_start|>0 0 1000 1000<|box_end|><|ref_start|>table<|ref_end|>"
    assert layout_regions(text, 100, 200, 1)[0]["bbox"] == [0, 0, 100, 200]
    with pytest.raises(ValueError, match="лимит"):
        layout_regions(text + text, 100, 200, 1)
    with pytest.raises(ValueError, match="диапазона"):
        layout_regions(text.replace("1000", "1001"), 100, 200, 2)
