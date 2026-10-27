import pytest
from pydantic import ValidationError

from ocr_service.backends import create_engine
from ocr_service.catalog import ARCHIVED_LABELS, LABELS, REMOVED_BACKENDS
from ocr_service.config import Settings
from ocr_service.schemas import PageResult
from ocr_service.vlm_models import artifacts


def test_only_four_providers_are_available_and_old_env_lists_still_work(client):
    expected = ["paddle", "dots_mocr", "glm_ocr", "lighton_small"]
    assert list(LABELS) == expected
    assert [item["id"] for item in client.get("/api/config").json()["backends"]] == expected
    assert set(client.get("/health").json()["workers"]) == set(expected)
    previous = (
        "paddle,dots_mocr,glm_ocr,chandra,lighton,kdl,lighton_small,jina,surya,falcon,nemotron"
    )
    assert Settings(_env_file=None, enabled_backends=previous).backends == expected
    with pytest.raises(ValidationError):
        Settings(_env_file=None, enabled_backends=previous + ",misspelled")


@pytest.mark.parametrize("backend", sorted(REMOVED_BACKENDS))
def test_removed_provider_cannot_run_but_its_archived_results_and_reports_remain_readable(
    client, settings, pdf_bytes, backend
):
    with pytest.raises(ValueError):
        create_engine(backend, settings)
    with pytest.raises(ValueError):
        artifacts(backend, settings)
    with pytest.raises(ValidationError):
        Settings(_env_file=None, worker_backend=backend)
    assert (
        client.post(
            "/api/documents", files={"file": ("document.pdf", pdf_bytes)}, data={"backend": backend}
        ).status_code
        == 422
    )
    assert (
        client.post(
            "/api/runs", json={"backend": backend, "filenames": ["document.pdf"]}
        ).status_code
        == 422
    )
    db = client.app.state.db
    old_id = "f" * 32
    db.create_job(old_id, "archive.pdf", "0" * 64, 1, {}, backend=backend)
    db.claim_job([backend])
    db.set_engine(old_id, {"name": ARCHIVED_LABELS[backend], "runtime": "historical"})
    db.save_page(old_id, PageResult(number=1, text="Архивный результат", seconds=1.5))
    db.finish_job(old_id)
    result = client.get(f"/api/documents/{old_id}/result")
    assert result.status_code == 200
    assert result.json()["document"]["configuration"]["label"] == ARCHIVED_LABELS[backend]
    assert result.json()["pages"][0]["text"] == "Архивный результат"
    run_id = client.get("/api/runs").json()["runs"][0]["id"]
    report = client.get(f"/api/runs/{run_id}/report?format=json")
    assert report.status_code == 200
    assert report.json()["documents"][0]["provider"] == ARCHIVED_LABELS[backend]
    assert report.json()["documents"][0]["pages"][0]["seconds"] == 1.5
