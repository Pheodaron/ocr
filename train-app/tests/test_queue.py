import json

import httpx
from fake_compute import FakeCompute
from sqlalchemy import select

from train_app.models import Job, Page, Result, Task
from train_app.queue import Dispatcher


async def test_bad_page_does_not_stop_next_page(client, app, page):
    with app.state.database() as session:
        first = session.get(Page, page["id"])
        session.add(
            Page(
                document_id=first.document_id,
                number=2,
                image_path=first.image_path,
                sha256=first.sha256,
                width=first.width,
                height=first.height,
            )
        )
        session.commit()
    job = client.post(
        f"/api/projects/{page['project_id']}/jobs", json={"kind": "lighton", "mode": "all"}
    ).json()
    dispatcher = Dispatcher(app.state.database, app.state.storage, app.state.settings)
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(FakeCompute(fail_first=True))
    ) as remote:
        with app.state.database() as session:
            stored = session.get(Job, job["id"])
        await dispatcher.execute(remote, stored)
        await dispatcher.execute(remote, stored)
    with app.state.database() as session:
        saved = session.get(Job, job["id"])
        assert saved.status == "completed" and saved.completed == 1 and saved.failed == 1
        assert len(session.scalars(select(Result)).all()) == 2
        assert not session.scalar(select(Job).where(Job.kind == "qwen"))


async def test_cancelled_queued_job_does_not_call_model(client, app, page):
    job = client.post(
        f"/api/projects/{page['project_id']}/jobs", json={"kind": "lighton", "mode": "all"}
    ).json()
    client.post(f"/api/projects/{page['project_id']}/jobs/{job['id']}/cancel")
    fake = FakeCompute()
    dispatcher = Dispatcher(app.state.database, app.state.storage, app.state.settings)
    async with httpx.AsyncClient(transport=httpx.MockTransport(fake)) as remote:
        assert await dispatcher.pick_job(remote, True) is None
    assert not fake.operations
    with app.state.database() as session:
        assert session.get(Job, job["id"]).status == "cancelled"
        assert all(
            task.status == "cancelled"
            for task in session.scalars(select(Task).where(Task.job_id == job["id"]))
        )


async def test_service_is_checked_once_for_many_waiting_jobs(client, app, page):
    for _ in range(3):
        response = client.post(
            f"/api/projects/{page['project_id']}/jobs", json={"kind": "lighton", "mode": "all"}
        )
        assert response.status_code == 202, response.text
    calls = []

    def unavailable(request):
        calls.append(request.url.path)
        return httpx.Response(503, json={"state": "loading"})

    dispatcher = Dispatcher(app.state.database, app.state.storage, app.state.settings)
    async with httpx.AsyncClient(transport=httpx.MockTransport(unavailable)) as remote:
        assert await dispatcher.pick_job(remote, True) is None
    assert calls == ["/status"]


async def test_reconnected_operation_uses_original_payload(client, app, page):
    job = client.post(
        f"/api/projects/{page['project_id']}/jobs", json={"kind": "lighton", "mode": "all"}
    ).json()
    sent = []

    def completed(request):
        if request.method == "PUT":
            sent.append(json.loads(request.content))
            return httpx.Response(202, json={"state": "completed"})
        return httpx.Response(200, json={"state": "completed", "result": {"ok": True}})

    dispatcher = Dispatcher(app.state.database, app.state.storage, app.state.settings)
    payload = {"project_id": page["project_id"], "history": ["original"]}
    async with httpx.AsyncClient(transport=httpx.MockTransport(completed)) as remote:
        await dispatcher.remote_operation(remote, "http://compute", "operation", payload, job["id"])
        await dispatcher.remote_operation(
            remote, "http://compute", "operation", {**payload, "history": ["changed"]}, job["id"]
        )
    assert sent == [payload, payload]
