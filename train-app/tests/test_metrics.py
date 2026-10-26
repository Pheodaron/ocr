import pytest

from train_app.metrics import compare, geometry_metrics
from train_app.tables import parse_table, validate_tables


def table(rows):
    return (
        "<table>"
        + "".join("<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" for row in rows)
        + "</table>"
    )


def test_missing_row_does_not_shift_numbers_or_remove_denominator():
    result = compare(
        table([["Код", "Сумма"], ["A", "0012"], ["B", "−0,25"], ["C", "30"]]),
        table([["Код", "Сумма"], ["A", "0012"], ["C", "30"]]),
    )
    assert result["numeric_cells"] == {"expected": 3, "predicted": 2, "correct": 2}
    assert result["tables"][0]["teds"] < 1 and not result["agrees"]


def test_missing_table_is_zero_quality():
    result = compare(table([["7"]]), "")
    assert result["missing_tables"] == 1 and result["tables"][0]["teds"] == 0
    assert result["numeric_cells"]["expected"] == 1


def test_cer_can_exceed_one_and_numbers_are_not_normalized():
    assert compare("а", "абвгд")["cer"] == 4
    result = compare(table([["0012"]]), table([["12"]]))
    assert result["numeric_cells"]["correct"] == 0 and not result["agrees"]


def test_headers_and_spans_are_measured():
    gold = '<table><tr><th colspan="2">Сумма</th></tr><tr><td>1</td><td>2</td></tr></table>'
    assert compare(gold, gold.replace("th", "td"))["tables"][0]["teds"] == 1
    assert compare(gold, table([["Сумма", ""], ["1", "2"]]))["tables"][0]["teds_structure"] < 1


def test_invalid_gold_and_unclosed_table_are_rejected():
    with pytest.raises(ValueError):
        parse_table('<table><tr><td rowspan="3">x</td></tr></table>')
    with pytest.raises(ValueError):
        validate_tables("<table><tr><td>1</td></tr>")


def test_extra_tables_prevent_agreement():
    result = compare(table([["1"]]), table([["1"]]) + table([["2"]]))
    assert result["extra_tables"] == 1 and not result["agrees"]


def test_geometry_penalizes_missing_regions_and_wrong_reading_order():
    gold = [
        {"bbox": [0, 0, 100, 100], "type": "title"},
        {"bbox": [0, 200, 100, 300], "type": "text"},
    ]
    reversed_result = geometry_metrics(gold, list(reversed(gold)))
    assert reversed_result["recall"] == 1
    assert reversed_result["mean_iou"] == 1
    assert not reversed_result["reading_order_correct"]
    missing = geometry_metrics(gold, gold[:1])
    assert missing["recall"] == 0.5 and not missing["reading_order_correct"]
