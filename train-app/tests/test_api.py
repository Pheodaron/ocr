import io
import zipfile

from sqlalchemy import select

from train_app.models import Checkpoint, Dataset, Job
from train_app.storage import atomic_json, read_json


def approve(client, page):
    return client.post(
        f"/api/projects/{page['project_id']}/pages/{page['id']}/annotations",
        json={
            "markdown": "Проверено",
            "blocks": [{"id": "text", "type": "text", "content": "Проверено", "bbox": None}],
            "text_approved": True,
        },
    )


def test_upload_does_not_start_ocr_and_projects_are_isolated(client, app, page):
    with app.state.database() as session:
        assert {job.kind for job in session.scalars(select(Job))} == {"ingest"}
    other = client.post("/api/projects", json={"title": "Другой"}).json()
    assert client.get(f"/api/projects/{other['id']}/pages/{page['id']}").status_code == 404
    assert client.get(f"/api/projects/{other['id']}/pages/{page['id']}/image").status_code == 404


def test_snapshot_is_immutable_and_unchecked_geometry_is_not_exported(client, page):
    annotation = approve(client, page).json()
    client.patch(
        f"/api/projects/{page['project_id']}/splits",
        json={"document_ids": [page["document_id"]], "split": "train"},
    )
    response = client.post(f"/api/projects/{page['project_id']}/datasets", json={"name": "Первая"})
    assert response.status_code == 201, response.text
    identifier = response.json()["id"]
    manifest = client.get(f"/api/projects/{page['project_id']}/datasets/{identifier}").json()
    assert manifest["items"][0]["grounding"] is None
    client.post(
        f"/api/projects/{page['project_id']}/pages/{page['id']}/annotations",
        json={
            "expected_revision": annotation["id"],
            "markdown": "Новая редакция",
            "blocks": [{"id": "text", "type": "text", "content": "Новая редакция", "bbox": None}],
            "text_approved": True,
        },
    )
    assert (
        client.get(f"/api/projects/{page['project_id']}/datasets/{identifier}").json()["items"][0][
            "markdown"
        ]
        == "Проверено"
    )
    download = client.get(f"/api/projects/{page['project_id']}/datasets/{identifier}/download")
    with zipfile.ZipFile(io.BytesIO(download.content)) as archive:
        assert "train.jsonl" in archive.namelist()
        assert '"prompt": "grounding"' not in archive.read("train.jsonl").decode()


def test_training_rejects_missing_validation(client, page):
    approve(client, page)
    client.patch(
        f"/api/projects/{page['project_id']}/splits",
        json={"document_ids": [page["document_id"]], "split": "train"},
    )
    dataset = client.post(
        f"/api/projects/{page['project_id']}/datasets", json={"name": "Только обучение"}
    ).json()
    response = client.post(
        f"/api/projects/{page['project_id']}/jobs",
        json={"kind": "train", "config": {"dataset_id": dataset["id"]}},
    )
    assert response.status_code == 422


def test_backup_restores_annotations_and_datasets(client, app, page):
    approve(client, page)
    project_id = page["project_id"]
    client.patch(
        f"/api/projects/{project_id}/splits",
        json={"document_ids": [page["document_id"]], "split": "test"},
    )
    dataset = client.post(f"/api/projects/{project_id}/datasets", json={"name": "Эталон"}).json()
    with app.state.database() as session:
        training = Job(project_id=project_id, kind="train", status="completed")
        session.add(training)
        session.flush()
        path = app.state.storage.project_dir(project_id) / "runs" / training.id / "checkpoint-1"
        atomic_json(path / "complete.json", {"step": 1})
        session.add(
            Checkpoint(
                project_id=project_id,
                job_id=training.id,
                path=app.state.storage.relative(path),
                step=1,
            )
        )
        session.commit()
    backup = client.get(f"/api/projects/{project_id}/backup")
    assert backup.status_code == 200, backup.text[:100]
    assert client.delete(f"/api/projects/{project_id}").status_code == 200
    restored = client.post(
        "/api/restore", files={"file": ("backup.zip", backup.content, "application/zip")}
    )
    assert restored.status_code == 201, restored.text
    assert (
        client.get(f"/api/projects/{project_id}/pages/{page['id']}").json()["annotation"][
            "text_approved"
        ]
        is True
    )
    with app.state.database() as session:
        saved = session.get(Dataset, dataset["id"])
        assert (
            read_json(app.state.storage.path(saved.manifest_path))["items"][0]["markdown"]
            == "Проверено"
        )
        checkpoint = session.scalar(select(Checkpoint).where(Checkpoint.project_id == project_id))
        assert read_json(app.state.storage.path(checkpoint.path) / "complete.json")["step"] == 1


def test_mutating_cross_origin_request_is_rejected(client):
    assert (
        client.post(
            "/api/projects", json={"title": "x"}, headers={"Origin": "https://example.org"}
        ).status_code
        == 403
    )
