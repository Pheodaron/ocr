import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from sqlalchemy import select

from train_app.config import Settings
from train_app.documents import render_page
from train_app.main import create_app
from train_app.models import Document, Job, Task


@pytest.fixture
def app(tmp_path):
    return create_app(
        Settings(data_dir=tmp_path / "data", frontend_dir=tmp_path / "missing", dispatcher=False)
    )


@pytest.fixture
def client(app):
    with TestClient(app) as client:
        yield client


@pytest.fixture
def project(client):
    response = client.post("/api/projects", json={"title": "Таблицы"})
    assert response.status_code == 201, response.text
    return response.json()


@pytest.fixture
def page(client, app, project):
    buffer = io.BytesIO()
    Image.new("RGB", (160, 220), "white").save(buffer, format="PNG")
    response = client.post(
        f"/api/projects/{project['id']}/documents",
        files={"file": ("таблица.png", buffer.getvalue(), "image/png")},
    )
    assert response.status_code == 202, response.text
    with app.state.database() as session:
        document = session.get(Document, response.json()["id"])
        rendered = render_page(document, 1, app.state.storage, app.state.settings)
        session.add(rendered)
        document.status = "ready"
        job = session.scalar(select(Job).where(Job.project_id == project["id"]))
        job.status, job.completed = "completed", 1
        task = session.scalar(select(Task).where(Task.job_id == job.id))
        task.status = "completed"
        session.commit()
        return {"id": rendered.id, "document_id": document.id, "project_id": project["id"]}
