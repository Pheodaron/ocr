import fcntl
import logging
import os
import signal
import subprocess
import sys
import threading
import time

from .config import Settings
from .db import Database
from .lifecycle import finalize_control

LOGGER = logging.getLogger(__name__)


def kill_inference(child) -> None:
    # Every document has a private session; include llama-server and any native
    # descendants, even when the Python child itself has already exited.
    try:
        os.killpg(child.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    child.wait()


def run_inference_child(settings: Settings, db: Database, job: dict, lock, stopped) -> int:
    started = time.monotonic()
    child = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "ocr_service.job_runner",
            job["id"],
            settings.model_dump_json(),
            str(os.getpid()),
        ],
        start_new_session=True,
        pass_fds=(lock.fileno(),),
    )
    shutdown_deadline = None
    try:
        while child.poll() is None:
            if db.cancellation_requested(job["id"]):
                LOGGER.info("Stopping inference for %s", job["id"])
                break
            if stopped.is_set():
                if shutdown_deadline is None:
                    child.terminate()
                    shutdown_deadline = time.monotonic() + 10
                elif time.monotonic() >= shutdown_deadline:
                    break
            time.sleep(0.2)
    finally:
        kill_inference(child)
        current = db.get_job(job["id"])
        if current and db.cancellation_requested(job["id"]):
            # A hard stop cannot run the child's final timing update. Account for
            # elapsed work without adding loading time or an already saved interval twice.
            loading = current["model_load_seconds"] - job["model_load_seconds"]
            recorded = current["processing_seconds"] - job["processing_seconds"]
            db.record_timings(
                job["id"], processing=max(0, time.monotonic() - started - loading - recorded)
            )
    return child.returncode


def run_worker(settings: Settings, engine_factory=None) -> None:
    settings.setup()
    db = Database(settings.db_path)
    db.initialize()
    stopped = threading.Event()
    state = {"state": "loading", "job_id": None, "error": None}
    backend = settings.worker_backend

    def report():
        db.heartbeat(**state, backend=backend)

    def on_signal(_signum, _frame):
        stopped.set()

    def heartbeat():
        while not stopped.wait(5):
            report()

    def finalize_controls():
        for requested in db.pending_controls(backend):
            try:
                finalize_control(db, settings, requested["id"])
            except OSError:
                LOGGER.exception("Could not delete document %s; will retry", requested["id"])

    lock = (settings.data_dir / f"worker-{backend}.lock").open("a")
    acquired = False
    try:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            acquired = True
        except BlockingIOError as error:
            raise RuntimeError(f"Для этого каталога уже запущен worker {backend}") from error
        signal.signal(signal.SIGTERM, on_signal)
        signal.signal(signal.SIGINT, on_signal)
        finalize_controls()
        db.recover(settings.max_page_attempts, [backend])
        report()
        thread = threading.Thread(target=heartbeat, name="heartbeat", daemon=True)
        thread.start()
        while not stopped.is_set():
            finalize_controls()
            state.update(state="idle", job_id=None, error=None)
            report()
            job = db.claim_job([backend])
            if job is None:
                stopped.wait(settings.poll_seconds)
                continue
            state.update(state="processing", job_id=job["id"])
            report()
            if engine_factory is not None:
                from .job_runner import run_document

                run_document(settings, job["id"], lambda _backend, config: engine_factory(config))
                continue
            # No Paddle/torch imports in this coordinator. Model memory dies with the child.
            returncode = run_inference_child(settings, db, job, lock, stopped)
            cancelled = db.cancellation_requested(job["id"])
            finalize_controls()
            if returncode and not cancelled and not stopped.is_set():
                LOGGER.error("Inference process exited with code %s", returncode)
                if (
                    not any(page.status == "processing" for page in db.pages(job["id"]))
                    and job["run_attempts"] >= settings.max_page_attempts
                ):
                    db.fail_job(job["id"], "Процесс OCR повторно завершился при загрузке модели")
            # Resume pages interrupted by a native crash; cap attempts to avoid a poison page.
            db.recover(settings.max_page_attempts, [backend])
    finally:
        stopped.set()
        if "thread" in locals():
            thread.join(timeout=6)
        # Preserve initialization errors for the client, but mark graceful shutdown.
        if acquired and state["state"] != "error":
            db.heartbeat("offline", backend=backend)
        lock.close()
