"""Strict OTSL table decoding: reject malformed merges instead of shifting cells."""

import html
import re


def otsl_to_html(content: str) -> str:
    content = content.strip().removeprefix("<otsl>").removesuffix("</otsl>")
    if "<otsl>" in content or "</otsl>" in content:
        raise ValueError("Вложенные OTSL-таблицы требуют ручной проверки")
    tokens = re.split(r"(<(?:fcel|ecel|lcel|ucel|xcel|nl)>)", content)
    grid = {}
    cells = []
    row = column = 0
    active = None
    for token in tokens:
        if token == "<nl>":
            if column == 0:
                raise ValueError("Пустая строка OTSL")
            row += 1
            column = 0
            active = None
        elif token in {"<fcel>", "<ecel>"}:
            if len(cells) >= 50000 or row > 10000 or column >= 1000:
                raise ValueError("Слишком большая OTSL-таблица")
            active = len(cells)
            cells.append({"row": row, "column": column, "text": ""})
            grid[row, column] = active
            column += 1
        elif token in {"<lcel>", "<ucel>", "<xcel>"}:
            left, above = grid.get((row, column - 1)), grid.get((row - 1, column))
            owner = left if token == "<lcel>" else above
            if owner is None or (token == "<xcel>" and left != above):
                raise ValueError("Некорректное объединение OTSL")
            grid[row, column] = owner
            column += 1
            active = None
        elif token.strip():
            if active is None:
                raise ValueError("Текст вне ячейки OTSL")
            cells[active]["text"] += token
    if not cells:
        raise ValueError("OTSL не содержит ячеек")
    positions = {index: [] for index in range(len(cells))}
    for position, owner in grid.items():
        positions[owner].append(position)
    rows = {}
    for index, cell in enumerate(cells):
        coordinates = positions[index]
        rowspan = max(r for r, _c in coordinates) - cell["row"] + 1
        colspan = max(c for _r, c in coordinates) - cell["column"] + 1
        if len(coordinates) != rowspan * colspan or rowspan > 200 or colspan > 200:
            raise ValueError("Непрямоугольное объединение OTSL")
        # Only a line break is meaningful markup inside OTSL cell text.
        text = html.escape(re.sub(r"<br\s*/?>", "\n", cell["text"], flags=re.I))
        markup = f'<td rowspan="{rowspan}" colspan="{colspan}">{text}</td>'
        rows.setdefault(cell["row"], []).append(markup)
    row_count = max(r for r, _c in grid) + 1
    return (
        "<table>"
        + "".join("<tr>" + "".join(rows.get(index, [])) + "</tr>" for index in range(row_count))
        + "</table>"
    )
