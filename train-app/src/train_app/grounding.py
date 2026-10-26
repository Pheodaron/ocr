"""Формат LightOnOCR-3: метка, нормированные координаты, содержимое."""

import re
from uuid import uuid4

MARKER = re.compile(
    r"!\[([\w+\-]+)\]\(\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,"
    r"\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\)"
)
LABELS = {
    "title",
    "section",
    "text",
    "list",
    "caption",
    "footnote",
    "aside_text",
    "header",
    "footer",
    "page_number",
    "table",
    "chart",
    "formula",
    "code",
    "image",
}


def block(content: str, label: str = "text", bbox=None) -> dict:
    return {"id": uuid4().hex, "type": label, "bbox": bbox, "content": content}


def parse_grounding(raw: str) -> tuple[str, list[dict], list[str]]:
    matches = list(MARKER.finditer(raw))
    if not matches:
        return raw.strip(), [block(raw.strip())], []
    blocks, warnings = [], []
    prefix = raw[: matches[0].start()].strip()
    if prefix:
        blocks.append(block(prefix))
    for index, marker in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(raw)
        box = [float(marker.group(i)) for i in range(2, 6)]
        if not valid_box(box):
            box = None
            warnings.append("Некорректная рамка: содержимое сохранено без координат")
        blocks.append(block(raw[marker.end() : end].strip(), marker.group(1), box))
    return join_blocks(blocks), blocks, warnings


def join_blocks(blocks: list[dict]) -> str:
    parts, previous_kept = [], False
    for item in blocks:
        text = item["content"].strip()
        if not text:
            previous_kept = False
            continue
        if item["type"].endswith("+") and previous_kept:
            parts[-1] += " " + text
        else:
            parts.append(text)
        previous_kept = True
    return "\n\n".join(parts)


def valid_box(box) -> bool:
    return (
        isinstance(box, list)
        and len(box) == 4
        and all(isinstance(value, (int, float)) and 0 <= value <= 1000 for value in box)
        and box[0] < box[2]
        and box[1] < box[3]
    )


def serialize_grounding(blocks: list[dict]) -> str:
    parts = []
    for item in blocks:
        if not valid_box(item.get("bbox")):
            raise ValueError("Для grounding нужны проверенные рамки каждого блока")
        label = item["type"]
        if label.removesuffix("+") not in LABELS:
            raise ValueError("Недопустимый тип блока")
        coords = ",".join(str(round(value)) for value in item["bbox"])
        parts.append(f"![{label}]({coords})\n{item['content'].strip()}")
    return "\n\n".join(parts)
