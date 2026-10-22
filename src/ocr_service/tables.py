"""Convert predicted HTML into a safe, loss-preserving table representation."""

from lxml import etree, html

from .schemas import Cell

ALLOWED_TAGS = {"table", "thead", "tbody", "tfoot", "tr", "td", "th", "caption", "br"}


def parse_table(markup: str) -> tuple[str, list[Cell]]:
    try:
        root = html.fromstring(markup, parser=html.HTMLParser(no_network=True))
    except etree.LxmlError as error:
        raise ValueError("Некорректная HTML-таблица") from error
    if root.tag != "table":
        found = root.xpath(".//table")
        if not found:
            raise ValueError("Результат модели не содержит HTML-таблицу")
        root = found[0]
    for node in list(root.iterdescendants()):
        if not isinstance(node.tag, str) or node.tag in {
            "script",
            "style",
            "iframe",
            "object",
            "img",
            "svg",
            "math",
        }:
            parent = node.getparent()
            if parent is not None:
                parent.remove(node)
        elif node.tag not in ALLOWED_TAGS:
            node.drop_tag()
    for node in root.iter():
        if not isinstance(node.tag, str):
            continue
        attributes = {}
        if node.tag in {"td", "th"}:
            for attr in ("rowspan", "colspan"):
                raw = node.get(attr, "1")
                try:
                    value = int(raw)
                except ValueError as error:
                    raise ValueError(f"Неверный {attr}") from error
                if value < 1 or value > 200:
                    raise ValueError(f"Слишком большой {attr}")
                if value > 1:
                    attributes[attr] = str(value)
        node.attrib.clear()
        node.attrib.update(attributes)
    cells = []
    occupied = set()
    rows = root.xpath("./tr | ./thead/tr | ./tbody/tr | ./tfoot/tr")
    if len(rows) > 10000:
        raise ValueError("Слишком много строк таблицы")
    for row_number, row in enumerate(rows):
        column = 0
        for node in row.xpath("./td | ./th"):
            while (row_number, column) in occupied:
                column += 1
            rowspan = int(node.get("rowspan", 1))
            colspan = int(node.get("colspan", 1))
            if column + colspan > 1000 or len(cells) >= 50000:
                raise ValueError("Слишком большая таблица")
            if len(occupied) + rowspan * colspan > 200000:
                raise ValueError("Слишком большая сетка объединённых ячеек")
            if any((row_number, c) in occupied for c in range(column, column + colspan)):
                raise ValueError("Перекрывающиеся объединения ячеек")
            cells.append(
                Cell(
                    row=row_number,
                    column=column,
                    text=" ".join(" ".join(node.itertext()).split()),
                    rowspan=rowspan,
                    colspan=colspan,
                    header=node.tag == "th",
                )
            )
            for r in range(row_number, row_number + rowspan):
                for c in range(column, column + colspan):
                    occupied.add((r, c))
            column += colspan
    if not cells:
        raise ValueError("HTML-таблица не содержит ячеек")
    return etree.tostring(root, encoding="unicode", method="html", with_tail=False), cells
