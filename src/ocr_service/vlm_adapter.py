"""Normalize generative OCR without inventing confidence or page coordinates."""

import json
import re

from lxml import html
from markdown_it import MarkdownIt

from .schemas import Block, PageResult, Table
from .tables import parse_table


def unfence(text: str) -> str:
    text = text.strip()
    if text.startswith("```") and text.endswith("```"):
        lines = text.splitlines()
        return "\n".join(lines[1:-1]).strip()
    return text


def adapt_vlm(raw: dict, number: int, dpi: int, width: int, height: int) -> PageResult:
    choice = raw["choices"][0]
    content = choice["message"].get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("VLM вернула пустой ответ")
    content = unfence(content)
    warnings = ["Генеративный OCR: оценка confidence отсутствует; проверьте текст и числа"]
    if choice.get("finish_reason") != "stop":
        warnings.append(f"Генерация не завершена штатно: {choice.get('finish_reason')}")
    blocks = []
    if raw["ocr_backend"] == "dots_mocr":
        try:
            layout = json.loads(content)
            if isinstance(layout, dict):
                layout = layout.get(
                    "layout", layout.get("blocks", layout.get("elements", [layout]))
                )
            if not isinstance(layout, list) or not layout:
                raise ValueError("Пустой layout")
            if len(layout) > 10000 or any(not isinstance(item, dict) for item in layout):
                raise ValueError("Некорректный layout")
            for item in layout:
                kind = str(item.get("category", "Text")).lower()
                text = item.get("text", "")
                if not isinstance(text, str):
                    raise ValueError("Неверный тип text в layout")
                blocks.append(Block(kind=kind, text=text))
            # llama.cpp can resize images inside its vision encoder. Retain generated boxes
            # in raw output; do not mislabel them as coordinates of our rendered preview.
            warnings.append("bbox dots.mocr сохранены в raw; координаты страницы не нормализованы")
            content = "\n\n".join(block.text for block in blocks if block.text)
        except (ValueError, TypeError) as error:
            # A truncated JSON is still valuable evidence, but cannot become fake cells.
            warnings.append(f"Не удалось разобрать layout JSON: {error}")
            blocks = []
    elif raw["ocr_backend"] in {"chandra", "surya"}:
        root = html.fragment_fromstring(
            content, create_parent="div", parser=html.HTMLParser(no_network=True)
        )
        for dangerous in root.xpath(".//script | .//style | .//iframe | .//object"):
            dangerous.drop_tree()
        for element in root.xpath(".//div[@data-bbox]"):
            bbox = normalized_bbox(element.get("data-bbox", "").split(), width, height)
            kind = element.get("data-label", "text").lower()
            blocks.append(Block(kind=kind, text="\n".join(element.itertext()).strip(), bbox=bbox))
        # HTML is never passed through to the client; tables are sanitized below.
        for element in root.xpath(".//table"):
            element.drop_tree()
        plain = "\n".join(root.itertext()).strip()
        original = html.fragment_fromstring(
            content, create_parent="div", parser=html.HTMLParser(no_network=True)
        )
        tables_markup = "\n\n".join(
            html.tostring(element, encoding="unicode") for element in original.xpath(".//table")
        )
        content = plain + ("\n\n" + tables_markup if tables_markup else "")
    elif raw["ocr_backend"] in {"lighton", "lighton_small"}:
        pattern = re.compile(r"!\[([\w+]+)\]\((\d+),(\d+),(\d+),(\d+)\)\s*")
        matches = list(pattern.finditer(content))
        for index, match in enumerate(matches):
            end = matches[index + 1].start() if index + 1 < len(matches) else len(content)
            blocks.append(
                Block(
                    kind=match[1].rstrip("+"),
                    text=content[match.end() : end].strip(),
                    bbox=normalized_bbox(match.groups()[1:], width, height),
                )
            )
    tables = []
    table_pattern = re.compile(r"<table\b[^>]*>.*?</table\s*>", re.IGNORECASE | re.DOTALL)
    for match in table_pattern.finditer(content):
        try:
            safe_html, cells = parse_table(match.group())
            tables.append(Table(html=safe_html, cells=cells))
        except ValueError as error:
            warnings.append(f"Некорректная HTML-таблица: {error}")
    # Some decoders emit pipe tables even when HTML is preferred. Decode them too.
    markdown = MarkdownIt("commonmark").enable("table")
    tokens = markdown.parse(content)
    index = 0
    while index < len(tokens):
        if tokens[index].type == "table_open":
            end = index + 1
            while end < len(tokens) and tokens[end].type != "table_close":
                end += 1
            if end < len(tokens):
                markup = markdown.renderer.render(tokens[index : end + 1], markdown.options, {})
                try:
                    safe_html, cells = parse_table(markup)
                    tables.append(Table(html=safe_html, cells=cells))
                except ValueError as error:
                    warnings.append(str(error))
                index = end
        index += 1
    if re.search(r"<table\b", content, re.I) and not tables:
        warnings.append("HTML-таблица не завершена или не разобрана")
    if raw.get("ocr_task") == "table" and not tables:
        warnings.append("Table Recognition не вернула корректную HTML-таблицу")
    # Preserve model Markdown verbatim. Plain text uses only sanitized table cells.
    replacements = {}
    for match in table_pattern.finditer(content):
        try:
            _markup, cells = parse_table(match.group())
            replacements[match.start()] = "\n".join(cell.text for cell in cells)
        except ValueError:
            pass

    def table_text(match):
        return replacements.get(match.start(), match.group())

    text = table_pattern.sub(table_text, content)
    # Do not parse arbitrary non-table HTML: it remains inert text in the client/export.
    return PageResult(
        number=number,
        width=width,
        height=height,
        dpi=dpi,
        text=text,
        markdown=content,
        blocks=blocks,
        tables=tables,
        confidence=None,
        needs_review=True,
        warnings=warnings,
        metrics={
            "usage": raw.get("usage", {}),
            "timings": raw.get("timings", {}),
            "finish_reason": choice.get("finish_reason"),
        },
    )


def normalized_bbox(values, width: int, height: int) -> list[float] | None:
    try:
        coordinates = [float(value) for value in values]
        if len(coordinates) != 4 or any(not 0 <= value <= 1000 for value in coordinates):
            return None
        x1, y1, x2, y2 = coordinates
        if x1 >= x2 or y1 >= y2:
            return None
        return [x1 * width / 1000, y1 * height / 1000, x2 * width / 1000, y2 * height / 1000]
    except (ValueError, TypeError):
        return None


def adapt_kdl(raw: dict, number: int, dpi: int, width: int, height: int) -> PageResult:
    blocks, tables, markdown = [], [], []
    warnings = list(raw["warnings"])
    for region in raw["blocks"]:
        text = region.get("text", "")
        kind = region["kind"]
        if region.get("table_html"):
            try:
                safe_html, cells = parse_table(region["table_html"])
                tables.append(Table(html=safe_html, cells=cells, bbox=region["bbox"]))
                markdown.append(safe_html)
                text = "\n".join(cell.text for cell in cells)
            except ValueError as error:
                warnings.append(str(error))
                markdown.append(text)
        else:
            markdown.append(text)
        blocks.append(Block(kind=kind, text=text, bbox=region["bbox"]))
    if all(region.get("error") for region in raw["blocks"]):
        raise ValueError("Не удалось распознать ни одну область KDL")
    calls = raw.get("calls", [])
    return PageResult(
        number=number,
        dpi=dpi,
        width=width,
        height=height,
        blocks=blocks,
        tables=tables,
        text="\n\n".join(block.text for block in blocks),
        markdown="\n\n".join(markdown),
        confidence=None,
        needs_review=True,
        warnings=warnings,
        metrics={
            "usage": {
                "completion_tokens": sum(call["completion_tokens"] for call in calls),
                "prompt_tokens": sum(call["prompt_tokens"] for call in calls),
            },
            "model_seconds": sum(call["seconds"] for call in calls),
            "calls": len(calls),
        },
    )
