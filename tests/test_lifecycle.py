import fcntl
import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

from ocr_service.schemas import PageResult


def upload(client, pdf_bytes, filename="document.pdf", backend="paddle"):
    return client.post(
        "/api/documents", files={"file": (filename, pdf_bytes)}, data={"backend": backend}
    ).json()


def wait_for(predicate, seconds=8):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.03)
    raise AssertionError("Timed out waiting for worker lifecycle transition")


def test_cancel_queued_job_never_runs_after_restart(client, pdf_bytes):
    job = upload(client, pdf_bytes)
    response = client.post(f"/api/documents/{job['id']}/cancel")
    assert response.status_code == 202
    assert response.json()["status"] == "cancelled"
    assert response.json()["cancelled_pages"] == 3
    db = client.app.state.db
    db.recover(2)
    assert db.claim_job() is None
    assert db.start_page(job["id"], 1) is None
    db.save_page(job["id"], PageResult(number=1, text="late result must not overwrite stop"))
    assert db.pages(job["id"])[0].status == "cancelled"
    assert client.post(f"/api/documents/{job['id']}/cancel").json()["status"] == "cancelled"
    for format in ("json", "markdown", "html"):
        assert client.get(f"/api/documents/{job['id']}/download?format={format}").status_code == 200


def test_cancel_stale_processing_without_worker_preserves_finished_pages(client, pdf_bytes):
    job = upload(client, pdf_bytes)
    db = client.app.state.db
    db.claim_job()
    db.save_page(job["id"], PageResult(number=1, text="saved before crash"))
    db.start_page(job["id"], 2)
    # Even a fresh stale heartbeat is not evidence of a live process/OS lock.
    db.heartbeat("processing", job_id=job["id"], backend="paddle")
    response = client.post(f"/api/documents/{job['id']}/cancel")
    assert response.json()["status"] == "cancelled"
    db.recover(2, ["paddle"])
    result = client.get(f"/api/documents/{job['id']}/result").json()
    assert result["pages"][0]["text"] == "saved before crash"
    assert [page["status"] for page in result["pages"]] == ["completed", "cancelled", "cancelled"]
    assert result["document"]["failed_pages"] == 0


@pytest.mark.parametrize(
    "status", ["queued", "processing", "failed", "partial", "completed", "cancelled"]
)
def test_delete_any_status_removes_files_pages_and_database(client, pdf_bytes, settings, status):
    job = upload(client, pdf_bytes)
    db = client.app.state.db
    with db.connect() as connection:
        connection.execute("UPDATE jobs SET status=? WHERE id=?", (status, job["id"]))
    directory = settings.data_dir / "documents" / job["id"]
    (directory / "page-1.raw.json").write_text("{}")
    response = client.delete(f"/api/documents/{job['id']}")
    assert response.status_code == 200
    assert response.json()["status"] == "deleted"
    assert not directory.exists() and db.get_job(job["id"]) is None
    assert not db.pages(job["id"])
    assert client.get(f"/api/documents/{job['id']}").status_code == 404
    assert client.get(f"/api/documents/{job['id']}/result").status_code == 404
    assert client.post(f"/api/documents/{job['id']}/cancel").status_code == 404
    assert client.get("/api/documents").json() == []


def test_delete_one_comparison_run_preserves_other_hardlinked_pdf(client, pdf_bytes, settings):
    comparison = client.post(
        "/api/comparisons",
        files={"file": ("same.pdf", pdf_bytes)},
        data={"backends": ["paddle", "dots_mocr"]},
    ).json()
    first, second = comparison["jobs"]
    assert client.delete(f"/api/documents/{first['id']}").status_code == 200
    remaining = client.get(f"/api/comparisons/{comparison['id']}").json()["jobs"]
    assert [job["id"] for job in remaining] == [second["id"]]
    assert (settings.data_dir / "documents" / second["id"] / "source.pdf").read_bytes() == pdf_bytes


def test_old_result_deletion_does_not_wait_for_another_document_on_same_backend(
    client, pdf_bytes, settings
):
    job = upload(client, pdf_bytes)
    db = client.app.state.db
    db.fail_job(job["id"], "old initialization failure")
    with (settings.data_dir / "worker-paddle.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert client.delete(f"/api/documents/{job['id']}").status_code == 200
    assert not (settings.data_dir / "documents" / job["id"]).exists()


@pytest.fixture
def lifecycle_worker(tmp_path, settings):
    # Real worker/coordinator + real OS child/session. The fixture substitutes
    # inference only; no model download or fake application backend is involved.
    script = tmp_path / "inference_fixture.py"
    script.write_text("""
import json, os, subprocess, sys, time
from pathlib import Path
from ocr_service.config import Settings
from ocr_service.db import Database
from ocr_service.lifecycle import arm_parent_death_signal
from ocr_service.schemas import PageResult
job_id, config, parent = sys.argv[1:]
arm_parent_death_signal(int(parent))
settings = Settings.model_validate_json(config)
db = Database(settings.db_path)
job = db.get_job(job_id)
directory = settings.data_dir / "documents" / job_id
if job["filename"] == "long.pdf":
    db.start_page(job_id, 1)
    db.save_page(job_id, PageResult(number=1, text="saved first page"))
    db.start_page(job_id, 2)
    descendant = subprocess.Popen([sys.executable, "-c",
        "import sys,time; from ocr_service.lifecycle import arm_parent_death_signal; "
        "arm_parent_death_signal(int(sys.argv[1])); time.sleep(60)", str(os.getpid())])
    # Include a descendant process and the real session identifier.
    children = [descendant.pid]
    (directory / "started.json").write_text(json.dumps({"pid":os.getpid(),"children":children}))
    while True: time.sleep(.1)
else:
    for page in db.pages(job_id):
        db.start_page(job_id, page.number)
        db.save_page(job_id, PageResult(number=page.number, text="next document completed"))
    db.finish_job(job_id)
""")
    launcher = """
import subprocess, sys
from ocr_service import worker
from ocr_service.config import Settings
original = subprocess.Popen
def fixture_inference(command, **kwargs):
    return original([sys.executable, sys.argv[1], command[3], command[4], command[5]], **kwargs)
worker.subprocess.Popen = fixture_inference
worker.run_worker(Settings.model_validate_json(sys.argv[2]))
"""
    processes = []

    def start():
        process = subprocess.Popen(
            [sys.executable, "-c", launcher, str(script), settings.model_dump_json()],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        processes.append(process)
        return process

    yield start
    for process in processes:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=12)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        output = process.communicate()[0]
        assert process.returncode == getattr(process, "expected_returncode", 0), output


def process_dead(pid):
    try:
        # A killed orphan may briefly remain a zombie until container init reaps it.
        return Path(f"/proc/{pid}/stat").read_text().split(")", 1)[1].strip().startswith("Z")
    except FileNotFoundError:
        return True


@pytest.mark.parametrize("action", ["cancel", "delete"])
def test_stop_real_process_and_descendant_frees_queue(
    client, pdf_bytes, settings, lifecycle_worker, action
):
    first = upload(client, pdf_bytes, "long.pdf")
    second = upload(client, pdf_bytes, "next.pdf")
    db = client.app.state.db
    lifecycle_worker()
    directory = settings.data_dir / "documents" / first["id"]
    marker = directory / "started.json"
    wait_for(marker.exists)
    processes = json.loads(marker.read_text())
    assert processes["children"]
    if action == "cancel":
        assert client.post(f"/api/documents/{first['id']}/cancel").status_code == 202
    else:
        assert client.delete(f"/api/documents/{first['id']}").status_code == 202
        assert client.get(f"/api/documents/{first['id']}").status_code == 404
    wait_for(lambda: db.get_job(second["id"])["status"] == "completed")
    wait_for(lambda: process_dead(processes["pid"]))
    for pid in processes["children"]:
        wait_for(lambda pid=pid: process_dead(pid))
    if action == "cancel":
        result = client.get(f"/api/documents/{first['id']}/result").json()
        assert result["document"]["status"] == "cancelled"
        assert result["pages"][0]["text"] == "saved first page"
        assert result["document"]["processing_seconds"] > 0
        assert [page["status"] for page in result["pages"]] == [
            "completed",
            "cancelled",
            "cancelled",
        ]
    else:
        assert db.get_job(first["id"]) is None
        assert not directory.exists()


def test_run_deletion_stops_real_child_and_preserves_the_next_run(
    client, pdf_bytes, settings, lifecycle_worker
):
    run = client.post(
        "/api/runs",
        json={
            "backend": "paddle",
            "filenames": ["long.pdf", "queued.pdf"],
        },
    ).json()
    jobs = []
    for index, name in enumerate(("long.pdf", "queued.pdf")):
        jobs.append(
            client.post(
                "/api/documents",
                files={"file": (name, pdf_bytes)},
                data={
                    "backend": "paddle",
                    "run_id": run["id"],
                    "file_index": index,
                },
            ).json()
        )
    other = upload(client, pdf_bytes, "next.pdf")
    db = client.app.state.db
    lifecycle_worker()
    marker = settings.data_dir / "documents" / jobs[0]["id"] / "started.json"
    wait_for(marker.exists)
    processes = json.loads(marker.read_text())
    response = client.delete(f"/api/runs/{run['id']}")
    assert response.status_code in {200, 202}
    assert client.get(f"/api/runs/{run['id']}").status_code == 404
    wait_for(lambda: db.get_job(other["id"])["status"] == "completed")
    wait_for(lambda: process_dead(processes["pid"]))
    for pid in processes["children"]:
        wait_for(lambda pid=pid: process_dead(pid))
    for job in jobs:
        assert db.get_job(job["id"]) is None
        assert not (settings.data_dir / "documents" / job["id"]).exists()
    assert not db.run_deletion_pending(run["id"])
    assert client.get(f"/api/documents/{other['id']}/result").status_code == 200


def test_stop_request_survives_recovery_and_is_finalized_on_worker_start(
    client, pdf_bytes, settings, lifecycle_worker
):
    first = upload(client, pdf_bytes, "long.pdf")
    db = client.app.state.db
    db.claim_job()
    db.start_page(first["id"], 1)
    with (settings.data_dir / "worker-paddle.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert client.post(f"/api/documents/{first['id']}/cancel").json()["status"] == "cancelling"
        db.recover(2, ["paddle"])
        assert db.get_job(first["id"])["status"] == "cancelling"
        assert db.claim_job(["paddle"]) is None
    second = upload(client, pdf_bytes, "next.pdf")
    lifecycle_worker()
    wait_for(lambda: db.get_job(second["id"])["status"] == "completed")
    assert db.get_job(first["id"])["status"] == "cancelled"
    assert db.pages(first["id"])[0].status == "cancelled"


def test_worker_crash_stops_child_and_allows_orphan_cancellation(
    client, pdf_bytes, settings, lifecycle_worker
):
    first = upload(client, pdf_bytes, "long.pdf")
    process = lifecycle_worker()
    marker = settings.data_dir / "documents" / first["id"] / "started.json"
    wait_for(marker.exists)
    inference = json.loads(marker.read_text())
    process.expected_returncode = -9
    process.kill()
    process.wait(timeout=3)
    wait_for(lambda: process_dead(inference["pid"]))
    for pid in inference["children"]:
        wait_for(lambda pid=pid: process_dead(pid))
    # The child's inherited flock is released; API can finalize without a worker.
    response = client.post(f"/api/documents/{first['id']}/cancel")
    assert response.json()["status"] == "cancelled"
    assert response.json()["successful_pages"] == 1
    assert client.get("/health").json()["workers"]["paddle"]["state"] == "offline"
