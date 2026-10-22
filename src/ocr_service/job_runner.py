"""Isolated document inference process. Exiting releases every native model allocation."""

import json
import logging
import signal
import sys
import threading
import time

from .backends import create_engine
from .config import Settings
from .db import Database
from .lifecycle import arm_parent_death_signal
from .processing import process_job


def run_document(settings: Settings, job_id: str, engine_factory=create_engine) -> None:
    db = Database(settings.db_path)
    job = db.get_job(job_id)
    if job is None or db.cancellation_requested(job_id):
        return
    settings = settings.restore_recognition_config(json.loads(job["config_json"]))
    stopped = threading.Event()

    def on_signal(_signum, _frame):
        stopped.set()

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    engine = None
    started = time.monotonic()
    try:
        try:
            engine = engine_factory(job["backend"], settings)
            db.record_timings(job_id, loading=time.monotonic() - started)
        except Exception as error:
            logging.exception("OCR initialization failed for %s", job["backend"])
            db.fail_job(job_id, f"Не удалось загрузить {job['backend']}: {error}")
            return
        signal.signal(signal.SIGTERM, on_signal)
        signal.signal(signal.SIGINT, on_signal)
        process_job(db, job, engine, settings, stopping=stopped.is_set)
    finally:
        if engine is not None and hasattr(engine, "close"):
            engine.close()


if __name__ == "__main__":
    if len(sys.argv) > 3:
        arm_parent_death_signal(int(sys.argv[3]))
    logging.basicConfig(level=logging.INFO)
    run_document(Settings.model_validate_json(sys.argv[2]), sys.argv[1])
