import copy
import os

from lxml import html

from ocr_service.engine import Prediction
from ocr_service.processing import process_job
from ocr_service.telemetry import ResourceMonitor, process_tree_snapshot


class ReportEngine:
    metadata = {
        "name": "test-fixture",
        "cpu_policy": {"threads": 2, "cpu_ids": [1, 2], "nice": 10, "quota": 3},
        "host": {"cpu_model": "CPU <script>alert(1)</script>"},
    }

    def __init__(self, prediction, fail_second=False):
        self.prediction = prediction
        self.calls = 0
        self.fail_second = fail_second

    def recognize(self, image, **kwargs):
        self.calls += 1
        if self.fail_second and self.calls in {2, 3}:
            raise ValueError("<script>failed page</script>")
        return Prediction(raw=copy.deepcopy(self.prediction), image=image)


def test_report_records_each_page_stages_resources_and_failed_passes(
    client, pdf_bytes, settings, prediction
):
    jobs = [
        client.post("/api/documents", files={"file": (name, pdf_bytes)}).json()
        for name in ("<script>.pdf", "second.pdf")
    ]
    db = client.app.state.db
    process_job(db, db.claim_job(), ReportEngine(prediction, fail_second=True), settings)
    ids = [item["id"] for item in jobs]
    before = client.get(f"/api/documents/{ids[0]}/result").json()
    response = client.post("/api/reports?format=json", json={"document_ids": ids + ids[:1]})
    assert response.status_code == 200
    assert "attachment" in response.headers["content-disposition"]
    report = response.json()
    assert not report["complete"]
    assert report["totals"]["documents"] == 2
    assert report["totals"]["pages"] == 6
    first, pending = report["documents"]
    assert [page["status"] for page in first["pages"]] == ["completed", "failed", "completed"]
    assert pending["pages"][0]["seconds"] is None
    assert first["engine"]["cpu_policy"]["threads"] == 2
    page = first["pages"][0]
    assert page["seconds"] > 0
    assert page["metrics"]["resources"]["sampled_tree_rss_peak_bytes"] > 0
    assert page["metrics"]["resources"]["cpu_seconds"] >= 0
    stages = page["passes"][0]["stages"]
    assert [stage["stage"] for stage in stages] == [
        "render_pdf",
        "save_input_image",
        "recognize",
        "save_raw_result",
        "parse_result",
    ]
    render = stages[0]
    assert render["details"]["dpi"] == settings.dpi
    assert render["details"]["width"] == page["width"]
    assert render["details"]["height"] == page["height"]
    assert all(stage["seconds"] >= 0 for stage in stages)
    failed = first["pages"][1]
    assert [attempt["dpi"] for attempt in failed["passes"]] == [settings.dpi, settings.retry_dpi]
    assert failed["passes"][0]["stages"][-1]["status"] == "failed"
    assert client.get(f"/api/documents/{ids[0]}/result").json() == before
    rendered = client.post("/api/reports?format=html", json={"document_ids": ids})
    assert rendered.status_code == 200
    root = html.fromstring(rendered.text)
    assert not root.xpath("//script")
    assert "<script>.pdf" in root.text_content()
    assert "Рендер страницы PDF" in root.text_content()
    assert "PyMuPDF" in root.text_content()
    assert "Промежуточный отчёт" in root.text_content()
    process_job(db, db.claim_job(), ReportEngine(prediction), settings)
    assert client.post("/api/reports?format=json", json={"document_ids": ids}).json()["complete"]


def test_status_snapshot_reports_missing_and_validates_ids(client, pdf_bytes):
    job = client.post("/api/documents", files={"file": ("test.pdf", pdf_bytes)}).json()
    response = client.post("/api/documents/status", json={"document_ids": [job["id"], "0" * 32]})
    assert response.status_code == 200
    assert response.json()["documents"][0]["id"] == job["id"]
    assert response.json()["missing"] == ["0" * 32]
    assert client.post("/api/documents/status", json={"document_ids": []}).status_code == 422
    assert client.post("/api/reports", json={"document_ids": ["invalid"]}).status_code == 422
    assert client.post("/api/reports", json={"document_ids": ["0" * 32]}).status_code == 404
    assert client.post("/api/reports", json={}).status_code == 422
    assert (
        client.post("/api/reports?format=pdf", json={"document_ids": [job["id"]]}).status_code
        == 422
    )


def test_upload_errors_are_included_even_without_accepted_documents(client):
    payload = {"upload_errors": [{"filename": "bad.pdf", "error": "<script>invalid PDF</script>"}]}
    response = client.post("/api/reports?format=json", json=payload)
    assert response.status_code == 200
    assert response.json()["totals"]["upload_errors"] == 1
    rendered = client.post("/api/reports?format=html", json=payload)
    root = html.fromstring(rendered.text)
    assert not root.xpath("//script")
    assert "bad.pdf" in root.text_content()
    assert "<script>invalid PDF</script>" in root.text_content()


def test_live_process_snapshot_includes_current_process():
    snapshot = process_tree_snapshot()
    own = [value for key, value in snapshot.items() if key[0] == os.getpid()]
    assert len(own) == 1
    assert own[0][0] >= 0 and own[0][1] > 0


def test_sampled_resource_deltas_include_new_children(monkeypatch):
    from ocr_service import telemetry

    monitor = ResourceMonitor()
    monitor.last = {(1, "start"): (10, 100)}
    monitor.cpu_seconds = 0
    monitor.rss_peak = 100
    monitor.samples = 1
    monkeypatch.setattr(
        telemetry,
        "process_tree_snapshot",
        lambda: {
            (1, "start"): (11, 120),
            (2, "child"): (0.5, 80),
        },
    )
    monitor._sample()
    assert monitor.cpu_seconds == 1.5
    assert monitor.rss_peak == 200
    assert monitor.metrics()["samples"] == 2
