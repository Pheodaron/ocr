import fcntl
import logging
import signal
import subprocess
import sys
import threading

from .config import Settings
from .db import Database

LOGGER = logging.getLogger(__name__)


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
        db.recover(settings.max_page_attempts, [backend])
        report()
        thread = threading.Thread(target=heartbeat, name="heartbeat", daemon=True)
        thread.start()
        while not stopped.is_set():
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
            child = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "ocr_service.job_runner",
                    job["id"],
                    settings.model_dump_json(),
                ]
            )
            signalled = False
            while child.poll() is None:
                if stopped.wait(0.5) and not signalled:
                    child.terminate()  # Child finishes its current page and closes llama-server.
                    signalled = True
                if signalled:
                    child.wait()
            if child.returncode:
                LOGGER.error("Inference process exited with code %s", child.returncode)
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
