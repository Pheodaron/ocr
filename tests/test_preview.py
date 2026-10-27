import pytest
from lxml import html

from ocr_service.preview import render_preview
from ocr_service.schemas import PageResult


def test_markdown_and_embedded_html_preserve_formatting_and_numbers():
    source = (
        "# Договор\n\n**Итого** и *примечание*\n\n- Первый\n- Второй\n\n"
        "| Код | Сумма |\n| --- | --- |\n| 0012 | −0,25 |\n\n"
        '<table><tr><th colspan="2">HTML</th></tr>'
        '<tr><td rowspan="2">001</td><td>2</td></tr><tr><td>3</td></tr></table>'
    )
    root = html.fragment_fromstring(render_preview(source), create_parent="div")
    assert root.find("h1").text == "Договор"
    assert root.find(".//strong").text == "Итого"
    assert len(root.findall(".//li")) == 2
    assert len(root.findall(".//table")) == 2
    assert root.xpath(".//td/text()") == ["0012", "−0,25", "001", "2", "3"]
    assert root.xpath(".//th[@colspan='2']/text()") == ["HTML"]
    assert root.xpath(".//td[@rowspan='2']/text()") == ["001"]


@pytest.mark.parametrize(
    "source",
    [
        '<script>alert(1)</script><p onclick="alert(1)">Текст</p>',
        '<svg onload="alert(1)"><foreignObject><img src=x onerror=alert(1)></foreignObject></svg>',
        '<iframe srcdoc="<script>alert(1)</script>"></iframe>'
        '<object data="/api/documents"></object>',
        '<style>body{display:none}</style><p style="position:fixed" id="job">Текст</p>',
        '<a href="javascript:alert(1)">Ссылка</a><a href="data:text/html,x">Данные</a>',
        '<a href="java&#10;script:alert(1)">Ссылка</a><img src="https://example.com/tracker">',
        '<form action="/api/documents"><button>Удалить</button></form>',
        "&lt;img src=x onerror=alert(1)&gt;",
        '<table><tr><td colspan="' + "9" * 5000 + '">Ячейка</td></tr></table>',
    ],
)
def test_preview_has_no_active_content_or_untrusted_attributes(source):
    root = html.fragment_fromstring(render_preview(source), create_parent="div")
    assert not root.xpath(".//script|.//style|.//svg|.//iframe|.//object|.//img|.//form|.//button")
    for element in root.iterdescendants():
        assert not any(name.startswith("on") for name in element.attrib)
        assert not element.get("style")
        assert not element.get("id")
        assert not element.get("href")


def test_plain_text_entities_and_code_remain_text():
    source = "Сумма < 5 &gt; 2 &amp; 1\n\n`<script>alert(1)</script>`\n\n```html\n<b>Код</b>\n```"
    root = html.fragment_fromstring(render_preview(source), create_parent="div")
    assert "Сумма < 5 > 2 & 1" in root.text_content()
    assert root.find(".//script") is None
    assert root.find(".//b") is None
    assert root.find(".//pre/code").text == "<b>Код</b>\n"


def test_external_links_open_separately_with_no_opener():
    root = html.fragment_fromstring(render_preview("[Ссылка](https://example.com)"))
    link = root.find("a")
    assert link.get("href") == "https://example.com"
    assert link.get("target") == "_blank"
    assert set(link.get("rel").split()) == {"noopener", "noreferrer"}


def test_page_preview_uses_original_markup_and_does_not_change_results(client, pdf_bytes):
    job = client.post("/api/documents", files={"file": ("document.pdf", pdf_bytes)}).json()
    path = f"/api/documents/{job['id']}"
    assert client.get(path + "/pages/1/preview").status_code == 409
    db = client.app.state.db
    db.claim_job()
    for number in range(1, 4):
        db.save_page(
            job["id"],
            PageResult(number=number, text="Обычный текст", markdown="# Исходная разметка"),
        )
    db.finish_job(job["id"])
    before = client.get(path + "/result").json()
    response = client.get(path + "/pages/1/preview")
    assert response.status_code == 200
    assert "<h1>Исходная разметка</h1>" in response.json()["html"]
    assert client.get(path + "/result").json() == before
    assert client.get(path + "/pages/4/preview").status_code == 404
    assert client.get(path + "/pages/0/preview").status_code == 422
    assert client.get("/api/documents/" + "0" * 32 + "/pages/1/preview").status_code == 404


def test_page_preview_falls_back_to_plain_text(client, pdf_bytes):
    job = client.post("/api/documents", files={"file": ("document.pdf", pdf_bytes)}).json()
    db = client.app.state.db
    db.claim_job()
    for number in range(1, 4):
        db.save_page(job["id"], PageResult(number=number, text="Текст без Markdown"))
    db.finish_job(job["id"])
    response = client.get(f"/api/documents/{job['id']}/pages/1/preview")
    assert response.status_code == 200
    assert "Текст без Markdown" in response.json()["html"]
