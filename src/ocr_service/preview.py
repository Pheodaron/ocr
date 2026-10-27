"""Render OCR Markdown and HTML without active content or external resources."""

import re
from html import escape

from lxml import etree, html
from markdown_it import MarkdownIt

ALLOWED_TAGS = {
    "p",
    "br",
    "hr",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "strong",
    "b",
    "em",
    "i",
    "s",
    "del",
    "u",
    "sub",
    "sup",
    "blockquote",
    "pre",
    "code",
    "ul",
    "ol",
    "li",
    "dl",
    "dt",
    "dd",
    "table",
    "thead",
    "tbody",
    "tfoot",
    "tr",
    "td",
    "th",
    "caption",
    "a",
    "div",
    "span",
}
DROP_TAGS = {
    "script",
    "style",
    "iframe",
    "object",
    "embed",
    "svg",
    "math",
    "template",
    "head",
    "title",
    "link",
    "meta",
    "base",
    "form",
    "input",
    "button",
    "textarea",
    "select",
    "img",
    "audio",
    "video",
    "source",
}


def render_preview(source: str) -> str:
    markup = MarkdownIt("commonmark", {"html": True}).enable("table").render(source)
    root = html.fragment_fromstring(markup, create_parent="div")
    for element in list(root.iterdescendants()):
        if element.getparent() is None:
            continue
        if not isinstance(element.tag, str):
            element.getparent().remove(element)
            continue
        if element.tag in DROP_TAGS:
            element.drop_tree()
        elif element.tag not in ALLOWED_TAGS:
            element.drop_tag()
        else:
            attributes = {}
            if element.tag in {"td", "th"}:
                for name in ("rowspan", "colspan"):
                    value = element.get(name, "")
                    if re.fullmatch(r"[0-9]{1,3}", value) and 1 <= int(value) <= 200:
                        attributes[name] = value
            if element.tag == "ol":
                start = element.get("start", "")
                if re.fullmatch(r"-?[0-9]{1,7}", start):
                    attributes["start"] = start
            if element.tag == "a":
                href = element.get("href", "")
                if re.match(r"^(https?://|mailto:)", href, re.I) and not any(
                    ord(char) <= 32 or ord(char) == 127 for char in href
                ):
                    attributes.update(href=href, target="_blank", rel="noopener noreferrer")
            element.attrib.clear()
            element.attrib.update(attributes)
    return escape(root.text or "") + "".join(
        etree.tostring(element, encoding="unicode", method="html") for element in root
    )
