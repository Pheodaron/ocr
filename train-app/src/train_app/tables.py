"""Проверка HTML и разбор объединённых ячеек без преобразования значений в числа."""

import re
from dataclasses import dataclass

from lxml import etree, html
from markdown_it import MarkdownIt

TABLE_PATTERN = re.compile(r"<table\b[^>]*>.*?</table\s*>", re.I | re.S)
MARKDOWN = MarkdownIt("commonmark", {"html": True}).enable("table")


@dataclass
class Cell:
    row: int
    column: int
    rowspan: int
    colspan: int
    text: str


@dataclass
class Table:
    element: etree._Element
    cells: list[Cell]
    rows: int
    columns: int


def normalized_text(text: str) -> str:
    return " ".join(text.split())


def parse_table(source: str) -> Table:
    parser = html.HTMLParser(no_network=True, recover=False)
    root = html.fragment_fromstring(source, create_parent="div", parser=parser)
    tables = root.xpath(".//table")
    if len(tables) != 1 or tables[0].xpath(".//table"):
        raise ValueError("Ожидалась одна таблица без вложенных таблиц")
    table = tables[0]
    rows = table.xpath("./tr | ./thead/tr | ./tbody/tr | ./tfoot/tr")
    if not rows:
        raise ValueError("В таблице нет строк")
    occupied: set[tuple[int, int]] = set()
    cells = []
    for row_number, row in enumerate(rows):
        column = 0
        for cell in row.xpath("./td | ./th"):
            while (row_number, column) in occupied:
                column += 1
            try:
                rowspan = int(cell.get("rowspan", "1"))
                colspan = int(cell.get("colspan", "1"))
            except ValueError as error:
                raise ValueError("Объединение ячеек должно быть целым числом") from error
            if not (1 <= rowspan <= len(rows) - row_number and 1 <= colspan <= 200):
                raise ValueError("Объединение ячеек выходит за границы таблицы")
            positions = {
                (r, c)
                for r in range(row_number, row_number + rowspan)
                for c in range(column, column + colspan)
            }
            if positions & occupied:
                raise ValueError("Объединённые ячейки пересекаются")
            occupied.update(positions)
            # br разделяет слова, но не теряет содержимое ячеек.
            copy = html.fromstring(etree.tostring(cell, encoding="unicode"))
            for line_break in copy.xpath(".//br"):
                line_break.tail = "\n" + (line_break.tail or "")
            cells.append(
                Cell(row_number, column, rowspan, colspan, normalized_text(copy.text_content()))
            )
            column += colspan
    width = max((column + 1 for _, column in occupied), default=0)
    if not cells:
        raise ValueError("В таблице нет ячеек")
    if any((row, column) not in occupied for row in range(len(rows)) for column in range(width)):
        raise ValueError("В таблице есть незаполненные места: добавьте пустые ячейки")
    return Table(table, cells, len(rows), width)


def extract_tables(markdown: str) -> list[str]:
    rendered = MARKDOWN.render(markdown)
    return [match.group() for match in TABLE_PATTERN.finditer(rendered)]


def validate_tables(markdown: str) -> None:
    # Незакрытый table браузер может исправить молча — эталон так принимать нельзя.
    if len(re.findall(r"<table\b", markdown, re.I)) != len(
        re.findall(r"</table\s*>", markdown, re.I)
    ):
        raise ValueError("Не закрыт HTML-элемент table")
    for source in extract_tables(markdown):
        parse_table(source)


def visible_text(markdown: str) -> str:
    rendered = MARKDOWN.render(markdown)
    document = html.fragment_fromstring(rendered, create_parent="div")
    for element in document.xpath(".//script | .//style"):
        element.drop_tree()
    for element in document.iter():
        if element.tag in {"br", "p", "tr", "td", "th", "div", "li", "h1", "h2", "h3"}:
            element.tail = " " + (element.tail or "")
    return normalized_text(document.text_content())
