import fcntl
import re
import time
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

from ocr_service.api import create_app
from ocr_service.catalog import LABELS
from ocr_service.config import Settings
from ocr_service.schemas import PageResult


@pytest.mark.parametrize("completed", [False, True])
def test_delete_run_removes_its_documents_and_files_but_preserves_other_runs(
    client, settings, pdf_bytes, completed
):
    run = client.post(
        "/api/runs",
        json={
            "backend": "paddle",
            "filenames": ["first.pdf", "bad.pdf", "last.pdf"],
        },
    ).json()
    ids = []
    for position in (0, 2):
        ids.append(
            client.post(
                "/api/documents",
                files={"file": ("document.pdf", pdf_bytes)},
                data={
                    "backend": "paddle",
                    "run_id": run["id"],
                    "file_index": position,
                },
            ).json()["id"]
        )
    client.post(
        f"/api/runs/{run['id']}/finish-upload",
        json={
            "failures": [
                {"position": 1, "error": "Invalid PDF"},
            ]
        },
    )
    db = client.app.state.db
    if completed:
        for job_id in ids:
            assert db.claim_job()["id"] == job_id
            for number in range(1, 4):
                db.save_page(job_id, PageResult(number=number, text="Saved result"))
            db.finish_job(job_id)
    other = client.post("/api/documents", files={"file": ("keep.pdf", pdf_bytes)}).json()
    settings.models_dir.mkdir(parents=True)
    weight = settings.models_dir / "weights.fixture"
    weight.write_bytes(b"preserved weights")
    response = client.delete(f"/api/runs/{run['id']}")
    assert response.status_code == 200
    assert response.json()["status"] == "deleted"
    for job_id in ids:
        assert db.get_job(job_id) is None
        assert not (settings.data_dir / "documents" / job_id).exists()
    assert client.get(f"/api/runs/{run['id']}").status_code == 404
    assert client.get(f"/api/runs/{run['id']}/report").status_code == 404
    assert client.delete(f"/api/runs/{run['id']}").status_code == 404
    assert client.get(f"/api/documents/{other['id']}").status_code == 200
    assert (settings.data_dir / "documents" / other["id"] / "source.pdf").read_bytes() == pdf_bytes
    assert weight.read_bytes() == b"preserved weights"
    db.initialize()
    assert len(client.get("/api/runs").json()["runs"]) == 1


def test_delete_run_with_no_accepted_files(client):
    run = client.post("/api/runs", json={"backend": "paddle", "filenames": ["bad.pdf"]}).json()
    response = client.delete(f"/api/runs/{run['id']}")
    assert response.status_code == 200
    assert client.get("/api/runs").json()["runs"] == []


def test_concurrent_file_removal_does_not_leave_an_orphan_directory(
    client, settings, pdf_bytes, monkeypatch
):
    from ocr_service import lifecycle

    job = client.post("/api/documents", files={"file": ("test.pdf", pdf_bytes)}).json()
    run_id = client.get("/api/runs").json()["runs"][0]["id"]
    directory = settings.data_dir / "documents" / job["id"]
    (directory / "race.raw.json").write_text("{}")
    (directory / "remaining.raw.json").write_text("{}")
    original = lifecycle.os.unlink

    def unlinked_by_another_deleter(path, *args, **kwargs):
        original(path, *args, **kwargs)
        if path == "race.raw.json":
            raise FileNotFoundError("Another cleanup already unlinked this entry")

    monkeypatch.setattr(lifecycle.os, "unlink", unlinked_by_another_deleter)
    assert client.delete(f"/api/runs/{run_id}").status_code == 200
    assert not directory.exists()
    assert client.app.state.db.get_job(job["id"]) is None


def test_upload_cannot_attach_a_new_document_after_run_deletion(
    client, pdf_bytes, settings, monkeypatch
):
    from ocr_service import api

    run = client.post("/api/runs", json={"backend": "paddle", "filenames": ["late.pdf"]}).json()
    original = api.validate_pdf

    def deleted_during_upload(path, max_pages):
        client.app.state.db.request_run_delete(run["id"])
        return original(path, max_pages)

    monkeypatch.setattr(api, "validate_pdf", deleted_during_upload)
    response = client.post(
        "/api/documents",
        files={"file": ("late.pdf", pdf_bytes)},
        data={
            "backend": "paddle",
            "run_id": run["id"],
            "file_index": 0,
        },
    )
    assert response.status_code == 409
    assert client.get("/api/documents").json() == []
    assert client.get("/api/runs").json()["runs"] == []
    assert not list((settings.data_dir / "documents").iterdir())


def test_pending_run_deletion_survives_restart_and_finishes_after_writer_stops(
    client, settings, pdf_bytes
):
    run = client.post("/api/runs", json={"backend": "paddle", "filenames": ["active.pdf"]}).json()
    job = client.post(
        "/api/documents",
        files={"file": ("active.pdf", pdf_bytes)},
        data={
            "backend": "paddle",
            "run_id": run["id"],
            "file_index": 0,
        },
    ).json()
    db = client.app.state.db
    db.claim_job()
    db.start_page(job["id"], 1)
    directory = settings.data_dir / "documents" / job["id"]
    with (settings.data_dir / "worker-paddle.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        response = client.delete(f"/api/runs/{run['id']}")
        assert response.status_code == 202
        assert response.json()["status"] == "deleting"
        assert db.get_job(job["id"])["delete_requested"]
        assert directory.exists()
        assert client.delete(f"/api/runs/{run['id']}").status_code == 202
        with TestClient(create_app(settings)) as restarted:
            assert restarted.get("/api/runs").json()["runs"] == []
            assert restarted.get(f"/api/documents/{job['id']}").status_code == 404
            assert db.claim_job() is None
            assert directory.exists()
    deadline = time.monotonic() + 6
    while db.get_job(job["id"]) is not None and time.monotonic() < deadline:
        time.sleep(0.05)
    assert db.get_job(job["id"]) is None
    assert not directory.exists()
    assert not db.run_deletion_pending(run["id"])


def test_existing_runs_gain_deletion_flag_without_losing_documents(client, pdf_bytes):
    job = client.post("/api/documents", files={"file": ("existing.pdf", pdf_bytes)}).json()
    db = client.app.state.db
    run_id = client.get("/api/runs").json()["runs"][0]["id"]
    with db.connect() as connection:
        connection.execute("DROP INDEX runs_deletions")
        connection.execute("ALTER TABLE runs DROP COLUMN delete_requested")
    db.initialize()
    assert client.get(f"/api/runs/{run_id}").json()["files"][0]["job_id"] == job["id"]
    assert client.delete(f"/api/runs/{run_id}").status_code == 200


def test_runs_preserve_groups_failures_reports_and_deleted_documents(client, settings, pdf_bytes):
    run = client.post(
        "/api/runs",
        json={
            "backend": "paddle",
            "filenames": ["first.pdf", "bad.pdf", "second.pdf"],
        },
    ).json()
    run_id = run["id"]
    ids = []
    for position in (0, 2):
        data = {"backend": "paddle", "run_id": run_id, "file_index": position}
        response = client.post("/api/documents", files={"file": ("doc.pdf", pdf_bytes)}, data=data)
        assert response.status_code == 202
        ids.append(response.json()["id"])
        # A retry after a lost response attaches no duplicate job or run.
        repeated = client.post("/api/documents", files={"file": ("doc.pdf", pdf_bytes)}, data=data)
        assert repeated.json()["id"] == ids[-1]
    wrong_provider = client.post(
        "/api/documents",
        files={"file": ("doc.pdf", pdf_bytes)},
        data={
            "backend": "glm_ocr",
            "run_id": run_id,
            "file_index": 1,
        },
    )
    assert wrong_provider.status_code == 409
    response = client.post(
        f"/api/runs/{run_id}/finish-upload",
        json={
            "failures": [
                {"position": 0, "error": "Lost response; already accepted"},
                {"position": 1, "error": "Повреждённый PDF"},
            ]
        },
    )
    assert response.status_code == 200
    files = response.json()["files"]
    assert [item["status"] for item in files] == ["accepted", "failed", "accepted"]
    assert files[0]["error"] is None
    assert [item["filename"] for item in files] == ["first.pdf", "bad.pdf", "second.pdf"]
    db = client.app.state.db
    for job_id in ids:
        assert db.claim_job()["id"] == job_id
        for number in range(1, 4):
            db.save_page(job_id, PageResult(number=number, text="Result", seconds=0.5))
        db.finish_job(job_id)
    with TestClient(create_app(settings)) as restarted:
        listing = restarted.get("/api/runs").json()
        assert len(listing["runs"]) == 1
        assert listing["runs"][0]["total"] == 3
        assert listing["runs"][0]["finished"] == 2
        report = restarted.get(f"/api/runs/{run_id}/report?format=json").json()
        assert report["run"]["id"] == run_id
        assert report["complete"]
        assert [item["document"]["id"] for item in report["documents"]] == ids
        assert report["upload_errors"][0]["filename"] == "bad.pdf"
        assert restarted.delete(f"/api/documents/{ids[0]}").status_code == 200
        detail = restarted.get(f"/api/runs/{run_id}").json()
        assert detail["files"][0]["status"] == "deleted"
        assert detail["files"][0]["job"] is None
        after = restarted.get(f"/api/runs/{run_id}/report?format=json").json()
        assert len(after["documents"]) == 1
        assert len(after["upload_errors"]) == 2


def test_run_pagination_validation_and_unfinished_uploads(client):
    ids = [
        client.post(
            "/api/runs",
            json={
                "backend": "paddle",
                "filenames": [f"file-{index}.pdf"],
            },
        ).json()["id"]
        for index in range(21)
    ]
    page = client.get("/api/runs?limit=20").json()
    assert len(page["runs"]) == 20 and page["has_more"]
    assert page["runs"][0]["accepted"] == page["runs"][0]["finished"] == 0
    assert client.get("/api/runs?limit=20&offset=20").json()["runs"][0]["id"] == ids[0]
    report = client.get(f"/api/runs/{ids[0]}/report?format=json").json()
    assert not report["complete"]
    finished = client.post(f"/api/runs/{ids[0]}/finish-upload", json={}).json()
    assert finished["files"][0]["status"] == "failed"
    assert (
        client.post("/api/runs", json={"backend": "jina", "filenames": ["a.pdf"]}).status_code
        == 422
    )
    assert client.post("/api/runs", json={"backend": "paddle", "filenames": []}).status_code == 422
    assert client.get("/api/runs/" + "0" * 32).status_code == 404


def test_legacy_job_migration_preserves_known_comparisons_without_duplicate_runs(client, pdf_bytes):
    comparison = client.post(
        "/api/comparisons",
        files={"file": ("a.pdf", pdf_bytes)},
        data={
            "backends": ["paddle", "glm_ocr"],
        },
    ).json()
    client.post("/api/documents", files={"file": ("b.pdf", pdf_bytes)})
    db = client.app.state.db
    with db.connect() as connection:
        connection.execute("DROP TABLE run_files")
        connection.execute("DROP TABLE runs")
    db.initialize()
    db.initialize()
    runs = db.runs()["runs"]
    assert len(runs) == 2
    assert db.run_details(comparison["id"])["backend"] == "comparison"
    assert len(db.run_details(comparison["id"])["files"]) == 2


def test_compose_services_are_explicit_and_environment_matches_the_model():
    path = Path(__file__).parents[1] / "compose.yaml"
    text = path.read_text()
    assert not any(
        isinstance(token, (yaml.tokens.AnchorToken, yaml.tokens.AliasToken))
        for token in yaml.scan(text)
    )
    compose = yaml.safe_load(text)
    assert not any(key.startswith("x-") for key in compose)
    services = compose["services"]
    assert len(services) == 1 + len(LABELS) * 2
    for name, service in services.items():
        assert "build" in service and "image" in service
        env = service["environment"]
        resolved = {
            key.removeprefix("OCR_").lower(): re.sub(r"\$\{[^:}]+:-([^}]+)\}", r"\1", str(value))
            for key, value in env.items()
        }
        Settings(_env_file=None, **resolved)
        if name == "api":
            continue
        backend = service["profiles"][0]
        assert env["OCR_ENABLED_BACKENDS"] == backend
        for prefix, owner in (
            ("OCR_DOTS_", "dots_mocr"),
            ("OCR_GLM_", "glm_ocr"),
            ("OCR_LIGHTON_SMALL_", "lighton_small"),
            ("OCR_KDL_", "kdl"),
        ):
            if backend != owner:
                assert not any(key.startswith(prefix) for key in env)
        if not name.startswith("models-"):
            assert env["OCR_WORKER_BACKEND"] == backend
            assert service["network_mode"] == "none"
            assert services["models-" + name]["build"]["target"] == "api"
