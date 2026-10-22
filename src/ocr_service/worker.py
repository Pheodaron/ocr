import fcntl
import logging
import signal
import threading

from .config import Settings
from .db import Database
from .processing import process_job

LOGGER = logging.getLogger(__name__)


def run_worker(settings: Settings, engine_factory=None) -> None:
    settings.setup()
    db = Database(settings.db_path)
    db.initialize()
    stopped = threading.Event()
    state = {"state": "loading", "job_id": None, "error": None}

    def on_signal(_signum, _frame):
        stopped.set()

    def heartbeat():
        while not stopped.wait(5):
            db.heartbeat(**state)

    lock = (settings.data_dir / "worker.lock").open("a")
    acquired = False
    try:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            acquired = True
        except BlockingIOError as error:
            raise RuntimeError("Для этого каталога данных уже запущен OCR worker") from error
        signal.signal(signal.SIGTERM, on_signal)
        signal.signal(signal.SIGINT, on_signal)
        db.recover(settings.max_page_attempts)
        db.heartbeat(**state)
        thread = threading.Thread(target=heartbeat, name="heartbeat", daemon=True)
        thread.start()
        try:
            if engine_factory is None:
                from .engine import PaddleEngine

                engine_factory = PaddleEngine
            engine = engine_factory(settings)
            # Native libraries may install their own handlers during import/startup.
            signal.signal(signal.SIGTERM, on_signal)
            signal.signal(signal.SIGINT, on_signal)
        except Exception as error:
            state.update(state="error", error=str(error)[:2000])
            db.heartbeat(**state)
            LOGGER.exception("OCR initialization failed; queued documents are preserved")
            raise
        while not stopped.is_set():
            state.update(state="idle", job_id=None, error=None)
            db.heartbeat(**state)
            job = db.claim_job()
            if job is None:
                stopped.wait(settings.poll_seconds)
                continue
            state.update(state="processing", job_id=job["id"])
            db.heartbeat(**state)
            process_job(db, job, engine, settings, stopping=stopped.is_set)
    finally:
        stopped.set()
        if "thread" in locals():
            thread.join(timeout=6)
        # Preserve initialization errors for the client, but mark graceful shutdown.
        if acquired and state["state"] != "error":
            db.heartbeat("offline")
        lock.close()
