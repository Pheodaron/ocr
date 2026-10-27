import re
from pathlib import Path

import yaml
from fastapi.testclient import TestClient

from ocr_service.api import create_app
from ocr_service.catalog import LABELS
from ocr_service.config import Settings
from ocr_service.schemas import PageResult


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
