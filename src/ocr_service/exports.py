import html

from .schemas import DocumentResult


def markdown_document(result: DocumentResult) -> str:
    sections = [f"# {result.document.filename}", f"Статус: {result.document.status}"]
    for page in result.pages:
        sections.append(f"## Страница {page.number}")
        if page.error:
            sections.append(f"Ошибка: {page.error}")
        else:
            sections.append(page.markdown)
        if page.warnings:
            sections.append("На проверку: " + "; ".join(page.warnings))
    return "\n\n".join(sections) + "\n"


def html_document(result: DocumentResult) -> str:
    parts = [
        '<!doctype html><html lang="ru"><meta charset="utf-8">',
        '<meta http-equiv="Content-Security-Policy" '
        "content=\"default-src 'none'; style-src 'unsafe-inline'\">",
        "<title>Результат OCR</title><style>body{font:16px system-ui;margin:2rem}"
        "table{border-collapse:collapse;margin:1rem 0}td,th{border:1px solid #999;"
        "padding:.4rem;white-space:pre-wrap}pre{white-space:pre-wrap}.warning{color:#8a4200}"
        "</style><body>",
        f"<h1>{html.escape(result.document.filename)}</h1>",
    ]
    for page in result.pages:
        parts.append(f"<section><h2>Страница {page.number}</h2>")
        if page.error:
            parts.append(f'<p class="warning">{html.escape(page.error)}</p>')
        else:
            for block in page.blocks:
                if block.kind != "table":
                    parts.append(f"<pre>{html.escape(block.text)}</pre>")
            if not page.blocks:
                parts.append(f"<pre>{html.escape(page.text)}</pre>")
            parts.extend(table.html for table in page.tables)
        for warning in page.warnings:
            parts.append(f'<p class="warning">{html.escape(warning)}</p>')
        parts.append("</section>")
    parts.append("</body></html>")
    return "\n".join(parts)
