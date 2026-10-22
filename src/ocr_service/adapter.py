import math
from statistics import mean

from .schemas import Block, PageResult, Table, TextFragment
from .tables import parse_table


def bbox(value) -> list[float]:
    if value is None:
        raise ValueError("Нет координат текстового фрагмента")
    if len(value) == 4 and all(isinstance(item, (int, float)) for item in value):
        return [float(item) for item in value]
    points = list(value)
    return [
        min(float(point[0]) for point in points),
        min(float(point[1]) for point in points),
        max(float(point[0]) for point in points),
        max(float(point[1]) for point in points),
    ]


def score(value) -> float | None:
    if value is None:
        return None
    number = float(value)
    return min(1.0, max(0.0, number)) if math.isfinite(number) else None


def adapt_prediction(
    raw: dict, number: int, dpi: int, width: int, height: int, threshold: float
) -> PageResult:
    result = raw.get("res", raw)
    page = PageResult(number=number, dpi=dpi, width=width, height=height)
    page.rotation = int(result.get("doc_preprocessor_res", {}).get("angle", 0))
    if page.rotation < 0:
        page.rotation = 0
    ocr = result.get("overall_ocr_res", {})
    texts = ocr.get("rec_texts", [])
    scores = ocr.get("rec_scores", [])
    boxes = ocr.get("rec_boxes", [])
    if not len(boxes):
        boxes = ocr.get("rec_polys", [])
    for index, text in enumerate(texts):
        if index >= len(boxes):
            page.warnings.append("У части OCR-текста отсутствуют координаты")
            break
        page.fragments.append(
            TextFragment(
                text=str(text),
                bbox=bbox(boxes[index]),
                confidence=score(scores[index]) if index < len(scores) else None,
            )
        )
    indexed = list(enumerate(result.get("parsing_res_list", [])))
    indexed.sort(
        key=lambda item: (
            item[1].get("block_order") is None,
            item[1].get("block_order") if item[1].get("block_order") is not None else item[0],
        )
    )
    for _, block in indexed:
        page.blocks.append(
            Block(
                kind=block.get("block_label", "unknown"),
                text=str(block.get("block_content", "")),
                bbox=bbox(block["block_bbox"]),
            )
        )
    table_blocks = [block for block in page.blocks if block.kind == "table"]
    predicted_tables = result.get("table_res_list", [])
    markups = [table.get("pred_html", "") for table in predicted_tables]
    # Parsing blocks can still contain tables when the separate table list is absent.
    if not markups:
        markups = [block.text for block in table_blocks]
    for markup in markups:
        try:
            safe_html, cells = parse_table(markup)
            matching_bbox = None
            for block in table_blocks:
                try:
                    if parse_table(block.text)[0] == safe_html:
                        matching_bbox = block.bbox
                        break
                except (ValueError, TypeError):
                    continue
            page.tables.append(Table(bbox=matching_bbox, html=safe_html, cells=cells))
        except (ValueError, TypeError) as error:
            page.warnings.append(f"Не удалось разобрать таблицу: {error}")
    parts = []
    plain_text = []
    for block in page.blocks:
        if block.kind == "table":
            try:
                safe_html, cells = parse_table(block.text)
                parts.append(safe_html)
                plain_text.append("\n".join(cell.text for cell in cells))
            except (ValueError, TypeError):
                continue
        else:
            plain_text.append(block.text)
            prefix = (
                "# "
                if block.kind == "doc_title"
                else ("## " if block.kind in {"paragraph_title", "title"} else "")
            )
            parts.append(prefix + block.text)
    page.text = "\n\n".join(plain_text) or "\n".join(item.text for item in page.fragments)
    page.markdown = "\n\n".join(parts) or page.text
    valid_scores = [item.confidence for item in page.fragments if item.confidence is not None]
    page.confidence = mean(valid_scores) if valid_scores else None
    if page.text.strip() and not valid_scores:
        page.warnings.append("Для распознанного текста отсутствуют оценки уверенности")
    if valid_scores and any(item < threshold for item in valid_scores):
        page.warnings.append("Есть текстовые фрагменты с низкой уверенностью OCR")
    if not page.text.strip():
        page.warnings.append("Текст не обнаружен: страница пустая или распознавание неполное")
    detected_tables = sum(
        box.get("label") == "table" for box in result.get("layout_det_res", {}).get("boxes", [])
    )
    if len(page.tables) < max(len(table_blocks), detected_tables):
        page.warnings.append("Не все найденные таблицы восстановлены")
    page.needs_review = bool(page.warnings)
    return page
