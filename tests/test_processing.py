import copy
import json

import pytest

from ocr_service.adapter import adapt_prediction
from ocr_service.catalog import LABELS
from ocr_service.engine import Prediction
from ocr_service.processing import process_job
from ocr_service.schemas import PageResult


class FakeEngine:
    """Only for service integration tests: never an application OCR backend."""

    metadata = {"name": "test-fixture"}

    def __init__(self, prediction):
        self.prediction = prediction
        self.calls = 0

    def recognize(self, image, **kwargs):
        self.calls += 1
        return Prediction(raw=copy.deepcopy(self.prediction), image=image)


def upload_and_claim(client, pdf_bytes):
    job = client.post("/api/documents", files={"file": ("test.pdf", pdf_bytes)}).json()
    return client.app.state.db, client.app.state.db.claim_job(), job["id"]


def test_page_failure_does_not_stop_later_pages(
    client, pdf_bytes, prediction, settings, monkeypatch
):
    from ocr_service import processing

    db, job, job_id = upload_and_claim(client, pdf_bytes)
    original = processing.render_page
    attempted_pages = []

    def renderer(source, number, dpi, max_pixels):
        attempted_pages.append(number)
        if number == 2:
            raise ValueError("bad page fixture")
        return original(source, number, dpi, max_pixels)

    monkeypatch.setattr(processing, "render_page", renderer)
    engine = FakeEngine(prediction)
    process_job(db, job, engine, settings)
    summary = db.summary(job_id)
    assert summary.status == "partial"
    assert (summary.successful_pages, summary.failed_pages, summary.processed_pages) == (2, 1, 3)
    assert 3 in attempted_pages
    result = client.get(f"/api/documents/{job_id}/result").json()
    assert [page["status"] for page in result["pages"]] == ["completed", "failed", "completed"]
    assert "bad page fixture" in result["pages"][1]["error"]
    assert result["pages"][2]["tables"][0]["cells"][1]["text"] == "150 кВт"
    image = client.get(result["pages"][2]["image_url"])
    assert image.status_code == 200
    assert image.content.startswith(b"\x89PNG")


def test_resume_preserves_finished_pages_and_limits_native_crash_retries(
    client, pdf_bytes, prediction, settings
):
    db, _job, job_id = upload_and_claim(client, pdf_bytes)
    db.save_page(job_id, PageResult(number=1, text="already saved"))
    db.start_page(job_id, 2)
    db.recover(settings.max_page_attempts)
    job = db.claim_job()
    db.start_page(job_id, 2)
    db.recover(settings.max_page_attempts)
    job = db.claim_job()
    engine = FakeEngine(prediction)
    process_job(db, job, engine, settings)
    assert engine.calls == 1
    assert db.pages(job_id)[0].text == "already saved"
    assert db.pages(job_id)[1].status == "failed"
    assert db.summary(job_id).status == "partial"


def test_failed_retry_preserves_first_pass(client, pdf_bytes, prediction, settings):
    db, job, job_id = upload_and_claim(client, pdf_bytes)
    prediction["res"]["overall_ocr_res"]["rec_scores"][0] = 0.4

    class RetryFailure(FakeEngine):
        def recognize(self, image, **kwargs):
            if image.shape[1] > 500:
                raise ValueError("retry failed")
            return super().recognize(image, **kwargs)

    process_job(db, job, RetryFailure(prediction), settings)
    result = db.pages(job_id)[0]
    assert result.status == "completed"
    assert result.needs_review
    assert result.dpi == settings.dpi
    assert len(result.passes) == 2
    assert result.passes[1]["error"] == "retry failed"


def test_adapter_preserves_unordered_footer_and_handles_missing_table_bbox(prediction):
    page = adapt_prediction(prediction, 1, 250, 1000, 1500, 0.75)
    assert page.text.startswith("Договор 00125")
    assert page.blocks[-1].kind == "footer"
    assert page.tables[0].bbox == [10, 60, 250, 90]
    assert not page.needs_review
    prediction["res"]["parsing_res_list"] = []
    page = adapt_prediction(prediction, 1, 250, 1000, 1500, 0.75)
    assert page.tables[0].bbox is None


def test_queue_claim_is_not_repeated(client, pdf_bytes):
    db, job, _job_id = upload_and_claim(client, pdf_bytes)
    assert job is not None
    assert db.claim_job() is None


def test_queued_document_keeps_submission_settings(client, pdf_bytes, prediction, settings):
    db, job, job_id = upload_and_claim(client, pdf_bytes)
    thresholds = []

    class RecordingEngine(FakeEngine):
        def recognize(self, image, **kwargs):
            thresholds.append(kwargs["layout_threshold"])
            return super().recognize(image, **kwargs)

    restarted_settings = settings.model_copy(update={"dpi": 200, "layout_threshold": 0.9})
    process_job(db, job, RecordingEngine(prediction), restarted_settings)
    assert all(page.dpi == settings.dpi for page in db.pages(job_id))
    assert thresholds == [settings.layout_threshold] * 3


def test_legacy_queued_job_keeps_previous_render_and_image_budget(
    client, pdf_bytes, settings, prediction
):
    from ocr_service.backends import configuration_details

    submitted = client.post(
        "/api/documents", files={"file": ("old.pdf", pdf_bytes)}, data={"backend": "dots_mocr"}
    ).json()
    db = client.app.state.db
    job = db.claim_job()
    legacy_config = json.loads(job["config_json"])
    legacy_config.pop("vlm_dpi")
    legacy_config.pop("dots_max_pixels")
    job["config_json"] = json.dumps(legacy_config)
    restored = settings.restore_recognition_config(legacy_config)
    assert restored.vlm_image_budget("dots_mocr") == 1_500_000
    assert configuration_details("dots_mocr", legacy_config)["render_dpi"] == settings.dpi
    process_job(db, job, FakeEngine(prediction), settings)
    assert all(page.dpi == settings.dpi for page in db.pages(submitted["id"]))


@pytest.mark.parametrize("backend", [name for name in LABELS if name != "paddle"])
def test_generative_backends_use_their_own_dpi_and_continue_after_error(
    client, pdf_bytes, settings, backend
):
    from ocr_service.backends import configuration_details

    submitted = client.post(
        "/api/documents", files={"file": ("test.pdf", pdf_bytes)}, data={"backend": backend}
    ).json()
    db = client.app.state.db
    shapes = []

    class Engine:
        metadata = {"name": "render-policy-fixture"}
        retry_on_review = False

        def recognize(self, image, **kwargs):
            shapes.append(image.shape)
            if len(shapes) == 2:
                raise ValueError("unreadable page fixture")
            return Prediction(
                raw={
                    "ocr_backend": "glm_ocr",
                    "choices": [{"message": {"content": "Document"}, "finish_reason": "stop"}],
                },
                image=image,
                format="vlm",
            )

    # Simulate a restarted worker with different defaults; the upload snapshot wins.
    restarted = settings.model_copy(update={"vlm_dpi": 200})
    process_job(db, db.claim_job(), Engine(), restarted)
    pages = db.pages(submitted["id"])
    assert [page.status for page in pages] == ["completed", "failed", "completed"]
    assert len(shapes) == 3  # errors also skip the higher-DPI pass
    assert shapes == [(625, 625, 3)] * 3  # 300 pt at the submitted 150 DPI
    assert pages[0].dpi == settings.vlm_dpi
    assert pages[2].dpi == settings.vlm_dpi
    details = configuration_details(backend, settings.recognition_config())
    assert details["render_dpi"] == settings.vlm_dpi and details["retry_dpi"] is None
