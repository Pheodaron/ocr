import json
import os
import sqlite3

from ocr_service.db import Database
from ocr_service.job_runner import run_document
from ocr_service.schemas import PageResult


def test_all_configurations_share_source_and_survive_restart(client, pdf_bytes, settings):
    names = [item["id"] for item in client.get("/api/config").json()["backends"]]
    from ocr_service.catalog import LABELS

    assert names == list(LABELS)
    response = client.post(
        "/api/comparisons", files={"file": ("test.pdf", pdf_bytes)}, data={"backends": names}
    )
    assert response.status_code == 202
    comparison = response.json()
    assert [job["backend"] for job in comparison["jobs"]] == names
    assert len({job["sha256"] for job in comparison["jobs"]}) == 1
    sources = [
        settings.data_dir / "documents" / job["id"] / "source.pdf" for job in comparison["jobs"]
    ]
    assert len({source.stat().st_ino for source in sources}) == 1
    db = Database(settings.db_path)
    db.initialize()
    claimed = db.claim_job()
    assert claimed["backend"] == "paddle"
    assert json.loads(claimed["config_json"])["dots_quantization"] == "Q5_K_M"
    db.fail_job(claimed["id"], "missing Paddle fixture")
    assert db.claim_job()["backend"] == "dots_mocr"
    assert (
        client.get(f"/api/comparisons/{comparison['id']}").json()["jobs"][0]["status"] == "failed"
    )


def test_backend_validation_and_batch_rollback(client, pdf_bytes, monkeypatch, settings):
    for backends in (["paddle", "unknown"], ["paddle", "paddle"]):
        assert (
            client.post(
                "/api/comparisons",
                files={"file": ("test.pdf", pdf_bytes)},
                data={"backends": backends},
            ).status_code
            == 422
        )
    assert (
        client.post(
            "/api/documents", files={"file": ("test.pdf", pdf_bytes)}, data={"backend": "unknown"}
        ).status_code
        == 422
    )
    monkeypatch.setattr(os, "link", lambda *_args: (_ for _ in ()).throw(OSError("link fixture")))
    # The entire batch is unpublished when file preparation fails.
    with __import__("pytest").raises(OSError, match="link fixture"):
        client.post(
            "/api/comparisons",
            files={"file": ("test.pdf", pdf_bytes)},
            data={"backends": ["paddle", "glm_ocr"]},
        )
    assert client.get("/api/documents").json() == []
    assert not list((settings.data_dir / "documents").iterdir())


def test_model_initialization_failure_is_terminal_for_one_job(client, pdf_bytes, settings):
    jobs = []
    for backend in ("dots_mocr", "glm_ocr"):
        jobs.append(
            client.post(
                "/api/documents", files={"file": ("test.pdf", pdf_bytes)}, data={"backend": backend}
            ).json()
        )
    db = client.app.state.db
    first = db.claim_job()

    def unavailable(_backend, _settings):
        raise RuntimeError("missing GGUF fixture")

    run_document(settings, first["id"], unavailable)
    assert db.summary(first["id"]).status == "failed"
    assert "missing GGUF fixture" in db.summary(first["id"]).error
    assert len(db.pages(first["id"])) == 3
    assert db.claim_job()["id"] == jobs[1]["id"]


def test_database_migrates_previous_release_without_losing_results(tmp_path):
    path = tmp_path / "old.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.execute("""CREATE TABLE jobs (
            id TEXT PRIMARY KEY, filename TEXT NOT NULL, sha256 TEXT NOT NULL,
            status TEXT NOT NULL, page_count INTEGER NOT NULL, created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL, error TEXT, engine_json TEXT NOT NULL DEFAULT '{}',
            config_json TEXT NOT NULL)""")
    db = Database(path)
    db.initialize()
    db.create_job("old", "old.pdf", "hash", 1, {})
    db.save_page("old", PageResult(number=1, text="previous result"))
    db.finish_job("old")
    db.initialize()
    assert db.summary("old").backend == "paddle"
    assert db.pages("old")[0].text == "previous result"


def test_separate_services_only_claim_and_recover_their_own_backend(client, pdf_bytes):
    for backend in ("dots_mocr", "glm_ocr"):
        client.post(
            "/api/documents", files={"file": ("test.pdf", pdf_bytes)}, data={"backend": backend}
        )
    db = client.app.state.db
    glm = db.claim_job(["glm_ocr"])
    dots = db.claim_job(["dots_mocr"])
    db.start_page(glm["id"], 1)
    db.start_page(dots["id"], 1)
    db.recover(2, ["glm_ocr"])
    assert db.get_job(glm["id"])["status"] == "queued"
    assert db.pages(glm["id"])[0].status == "pending"
    assert db.get_job(dots["id"])["status"] == "processing"
    assert db.pages(dots["id"])[0].status == "processing"
    assert db.claim_job(["paddle"]) is None
    assert db.claim_job(["dots_mocr"]) is None


def test_health_reports_each_service_independently(client):
    db = client.app.state.db
    db.heartbeat("processing", backend="glm_ocr")
    db.heartbeat("offline", backend="dots_mocr")
    result = client.get("/health").json()
    assert result["workers"]["glm_ocr"]["ready"]
    assert not result["workers"]["dots_mocr"]["ready"]
    assert not result["workers"]["kdl"]["ready"]
    assert client.get("/ready").status_code == 200
    db.heartbeat("offline", backend="glm_ocr")
    assert client.get("/ready").status_code == 503
