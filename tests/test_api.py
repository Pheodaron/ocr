import json
from datetime import UTC, datetime, timedelta

import pymupdf

from ocr_service.schemas import PageResult


def test_pdf_upload_validation_and_async_contract(client, pdf_bytes, settings):
    response = client.post("/api/documents", files={"file": ("договор.pdf", pdf_bytes)})
    assert response.status_code == 202
    job = response.json()
    assert job["page_count"] == 3
    assert job["status"] == "queued"
    assert job["filename"] == "договор.pdf"
    assert client.get(f"/api/documents/{job['id']}/result").status_code == 409
    assert client.get(f"/api/documents/{job['id']}").json()["processed_pages"] == 0
    assert (settings.data_dir / "documents" / job["id"] / "source.pdf").read_bytes() == pdf_bytes
    assert client.get("/api/documents").json()[0]["id"] == job["id"]


def test_reject_bad_empty_password_and_excessive_pdf(client, settings):
    assert client.post("/api/documents", files={"file": ("bad.pdf", b"garbage")}).status_code == 422
    assert client.post("/api/documents", files={"file": ("empty.pdf", b"")}).status_code == 422
    document = pymupdf.open()
    for _ in range(settings.max_pages + 1):
        document.new_page()
    assert (
        client.post("/api/documents", files={"file": ("long.pdf", document.tobytes())}).status_code
        == 422
    )
    encrypted = document.tobytes(
        encryption=pymupdf.PDF_ENCRYPT_AES_256, owner_pw="owner", user_pw="secret"
    )
    assert (
        client.post("/api/documents", files={"file": ("locked.pdf", encrypted)}).status_code == 422
    )
    assert (
        client.post(
            "/api/documents", files={"file": ("large.pdf", b"x" * (2 * 1024**2))}
        ).status_code
        == 413
    )
    assert client.get("/api/documents").json() == []
    assert not list((settings.data_dir / "documents").iterdir())


def test_result_errors_are_downloadable_and_html_is_escaped(client, pdf_bytes):
    job = client.post("/api/documents", files={"file": ("<script>.pdf", pdf_bytes)}).json()
    db = client.app.state.db
    db.claim_job()
    for number in range(1, 4):
        db.save_page(
            job["id"],
            PageResult(
                number=number, status="failed", error="<script>alert(1)</script>", needs_review=True
            ),
        )
    db.finish_job(job["id"])
    result = client.get(f"/api/documents/{job['id']}/result").json()
    assert result["document"]["status"] == "failed"
    for format in ("json", "markdown", "html"):
        response = client.get(f"/api/documents/{job['id']}/download?format={format}")
        assert response.status_code == 200
        assert "attachment" in response.headers["content-disposition"]
        if format == "json":
            assert json.loads(response.text)["schema_version"] == "1.0"
        if format == "html":
            assert "<script>" not in response.text
            assert "&lt;script&gt;" in response.text


def test_worker_readiness_includes_stale_and_initialization_errors(client):
    assert client.get("/health").status_code == 200
    assert client.get("/ready").status_code == 503
    db = client.app.state.db
    db.heartbeat("loading")
    assert not client.get("/health").json()["worker"]["ready"]
    db.heartbeat("idle")
    assert client.get("/ready").status_code == 200
    with db.connect() as connection:
        connection.execute(
            "UPDATE worker SET heartbeat=?",
            ((datetime.now(UTC) - timedelta(minutes=5)).isoformat(),),
        )
    assert client.get("/health").json()["worker"]["state"] == "offline"
    db.heartbeat("error", error="missing models")
    assert client.get("/ready").json()["worker"]["error"] == "missing models"


def test_local_client_and_docs_do_not_need_external_assets(client):
    for path in ("/", "/docs", "/static/app.js", "/static/style.css", "/openapi.json"):
        response = client.get(path)
        assert response.status_code == 200
        assert "cdn.jsdelivr" not in response.text
    assert client.get("/api/documents/not-an-id").status_code == 422
    assert client.get("/api/documents/" + "a" * 32).status_code == 404
