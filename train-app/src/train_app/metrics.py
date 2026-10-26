"""Воспроизводимые показатели: CER, TEDS, строки и точные числовые ячейки."""

import re
from dataclasses import dataclass

from apted import APTED, Config
from rapidfuzz.distance import Levenshtein

from .tables import Table, extract_tables, normalized_text, parse_table, visible_text

MATCHER_VERSION = "ordered-dp-1"
NUMBER = re.compile(r"\d")


def geometry_metrics(expected: list[dict], predicted: list[dict], threshold=0.5) -> dict:
    """Однозначное сопоставление по IoU; текстовое совпадение не подменяет положение."""

    def iou(left, right):
        intersection = max(0, min(left[2], right[2]) - max(left[0], right[0])) * max(
            0, min(left[3], right[3]) - max(left[1], right[1])
        )
        area = (left[2] - left[0]) * (left[3] - left[1]) + (right[2] - right[0]) * (
            right[3] - right[1]
        )
        return intersection / (area - intersection) if area > intersection else 0

    gold = [(i, block) for i, block in enumerate(expected) if block.get("bbox")]
    answers = [(i, block) for i, block in enumerate(predicted) if block.get("bbox")]
    candidates = sorted(
        [(iou(a["bbox"], b["bbox"]), i, j) for i, a in gold for j, b in answers], reverse=True
    )
    used_gold, used_answers, matches = set(), set(), []
    for overlap, left, right in candidates:
        if overlap >= threshold and left not in used_gold and right not in used_answers:
            used_gold.add(left)
            used_answers.add(right)
            matches.append(
                {
                    "expected": left,
                    "predicted": right,
                    "iou": overlap,
                    "type_equal": expected[left]["type"] == predicted[right]["type"],
                }
            )
    ordered = sorted(matches, key=lambda item: item["expected"])
    return {
        "matcher": "greedy-iou-1",
        "threshold": threshold,
        "matches": matches,
        "expected": len(gold),
        "predicted": len(answers),
        "matched": len(matches),
        "precision": len(matches) / len(answers) if answers else 0,
        "recall": len(matches) / len(gold) if gold else None,
        "mean_iou": sum(item["iou"] for item in matches) / len(matches) if matches else 0,
        "correct_types": sum(item["type_equal"] for item in matches),
        "reading_order_correct": len(matches) == len(gold)
        and all(
            a["predicted"] < b["predicted"] for a, b in zip(ordered, ordered[1:], strict=False)
        ),
    }


def similarity(left: str, right: str) -> float:
    return Levenshtein.normalized_similarity(left, right)


def align(left: list, right: list, score, gap: float = 0.45) -> list[tuple[int | None, int | None]]:
    """Выравнивание с сохранением порядка; пропуски остаются явными элементами."""
    values = [[0.0] * (len(right) + 1) for _ in range(len(left) + 1)]
    moves = {}
    for i in range(1, len(left) + 1):
        values[i][0] = -i * gap
        moves[i, 0] = (i - 1, 0)
    for j in range(1, len(right) + 1):
        values[0][j] = -j * gap
        moves[0, j] = (0, j - 1)
    for i in range(1, len(left) + 1):
        for j in range(1, len(right) + 1):
            choices = [
                (values[i - 1][j - 1] + 2 * score(left[i - 1], right[j - 1]) - 1, i - 1, j - 1),
                (values[i - 1][j] - gap, i - 1, j),
                (values[i][j - 1] - gap, i, j - 1),
            ]
            value, previous_i, previous_j = max(choices, key=lambda item: item[0])
            values[i][j] = value
            moves[i, j] = previous_i, previous_j
    pairs, i, j = [], len(left), len(right)
    while i or j:
        previous_i, previous_j = moves[i, j]
        pairs.append((i - 1 if previous_i < i else None, j - 1 if previous_j < j else None))
        i, j = previous_i, previous_j
    return list(reversed(pairs))


@dataclass
class Node:
    tag: str
    text: str
    spans: tuple[int, int]
    children: list


def tree(table: Table, structure_only: bool) -> Node:
    rows = []
    for row in range(table.rows):
        cells = [
            Node("td", "" if structure_only else cell.text, (cell.rowspan, cell.colspan), [])
            for cell in table.cells
            if cell.row == row
        ]
        rows.append(Node("tr", "", (1, 1), cells))
    return Node("table", "", (1, 1), rows)


class TableDistance(Config):
    def rename(self, left, right):
        if left.tag != right.tag or left.spans != right.spans:
            return 1
        return 1 - similarity(left.text, right.text) if left.tag == "td" else 0


def teds(left: Table, right: Table, structure_only=False) -> float:
    distance = APTED(
        tree(left, structure_only), tree(right, structure_only), TableDistance()
    ).compute_edit_distance()
    size = max(1 + left.rows + len(left.cells), 1 + right.rows + len(right.cells))
    return max(0.0, 1 - distance / size)


def row_cells(table: Table) -> list[list]:
    return [[cell for cell in table.cells if cell.row == row] for row in range(table.rows)]


def numeric_cells(left: Table, right: Table) -> dict:
    expected = [cell for cell in left.cells if NUMBER.search(cell.text)]
    predicted = [cell for cell in right.cells if NUMBER.search(cell.text)]
    row_pairs = align(
        row_cells(left),
        row_cells(right),
        lambda a, b: similarity(" | ".join(c.text for c in a), " | ".join(c.text for c in b)),
    )
    mapping = {a: b for a, b in row_pairs if a is not None and b is not None}
    correct = 0
    for cell in expected:
        candidates = [
            c
            for c in predicted
            if c.row == mapping.get(cell.row)
            and (c.column, c.rowspan, c.colspan) == (cell.column, cell.rowspan, cell.colspan)
        ]
        correct += bool(candidates and candidates[0].text == cell.text)
    return {"expected": len(expected), "predicted": len(predicted), "correct": correct}


def compare(expected: str, predicted: str) -> dict:
    left_text, right_text = visible_text(expected), visible_text(predicted)
    edits = Levenshtein.distance(left_text, right_text)
    result = {
        "matcher_version": MATCHER_VERSION,
        "cer": edits / len(left_text) if left_text else None,
        "character_edits": edits,
        "reference_characters": len(left_text),
        "blank_reference_extra_text": bool(not left_text and right_text),
        "text_equal": left_text == right_text,
        "tables": [],
        "prediction_errors": [],
    }
    # Ошибка эталона — ошибка расчёта, а не нулевое качество модели.
    gold = [parse_table(source) for source in extract_tables(expected)]
    answers = []
    for source in extract_tables(predicted):
        try:
            answers.append(parse_table(source))
        except ValueError as error:
            result["prediction_errors"].append(str(error))
    pairs = align(
        gold,
        answers,
        lambda a, b: similarity(
            " | ".join(c.text for c in a.cells), " | ".join(c.text for c in b.cells)
        ),
    )
    numbers = {"expected": 0, "predicted": 0, "correct": 0}
    missing = extra = exact = 0
    for a, b in pairs:
        left = gold[a] if a is not None else None
        right = answers[b] if b is not None else None
        if left is None:
            extra += 1
            n = numeric_cells(right, right)
            numbers["predicted"] += n["predicted"]
            continue
        if right is None:
            missing += 1
            n = numeric_cells(left, left)
            numbers["expected"] += n["expected"]
            result["tables"].append(
                {"expected": a, "predicted": None, "teds": 0, "teds_structure": 0, "exact": False}
            )
            continue
        n = numeric_cells(left, right)
        for key in numbers:
            numbers[key] += n[key]
        full = teds(left, right)
        correct = full == 1
        exact += correct
        result["tables"].append(
            {
                "expected": a,
                "predicted": b,
                "teds": full,
                "teds_structure": teds(left, right, True),
                "exact": correct,
            }
        )
    result.update(
        {
            "missing_tables": missing,
            "extra_tables": extra,
            "reference_tables": len(gold),
            "predicted_tables": len(answers),
            "exact_tables": exact,
            "numeric_cells": numbers,
        }
    )
    result["agrees"] = (
        normalized_text(expected) == normalized_text(predicted)
        or (result["text_equal"] and not missing and not extra and exact == len(gold))
    ) and not result["prediction_errors"]
    return result


def aggregate(pages: list[dict]) -> dict:
    metrics = [page["metrics"] for page in pages if page.get("metrics")]
    chars = sum(m["reference_characters"] for m in metrics)
    tables = [table for m in metrics for table in m["tables"]]
    numeric = {
        key: sum(m["numeric_cells"][key] for m in metrics)
        for key in ("expected", "predicted", "correct")
    }
    return {
        "pages": len(pages),
        "failed_pages": sum(bool(p.get("error")) for p in pages),
        "cer": sum(m["character_edits"] for m in metrics) / chars if chars else None,
        "teds": sum(t["teds"] for t in tables) / len(tables) if tables else None,
        "teds_structure": sum(t["teds_structure"] for t in tables) / len(tables)
        if tables
        else None,
        "exact_tables": sum(m["exact_tables"] for m in metrics),
        "reference_tables": sum(m["reference_tables"] for m in metrics),
        "missing_tables": sum(m["missing_tables"] for m in metrics),
        "extra_tables": sum(m["extra_tables"] for m in metrics),
        "numeric_cells": numeric,
        "numeric_accuracy": numeric["correct"] / numeric["expected"]
        if numeric["expected"]
        else None,
    }
