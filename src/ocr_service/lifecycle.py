"""Durable stop/delete requests shared by the API and isolated workers."""

import ctypes
import fcntl
import os
import shutil
import signal

from .config import Settings
from .db import Database
from .schemas import TERMINAL_STATUSES


def ignore_missing_during_cleanup(_function, _path, error) -> None:
    # A worker and the API may finish the same deletion concurrently. Continue
    # with the remaining files instead of abandoning a partially removed directory.
    if not isinstance(error, FileNotFoundError):
        raise error


def arm_parent_death_signal(parent: int) -> None:
    # Run in the new Python interpreter, not preexec_fn in a multithreaded worker.
    ctypes.CDLL(None).prctl(1, signal.SIGKILL)
    if os.getppid() != parent:
        os.kill(os.getpid(), signal.SIGKILL)


def finalize_control(db: Database, settings: Settings, job_id: str) -> None:
    """Caller has stopped inference, or the job is terminal and cannot write again."""
    db.finalize_cancel(job_id)
    job = db.get_job(job_id)
    if job and job["delete_requested"]:
        directory = settings.data_dir / "documents" / job_id
        try:
            shutil.rmtree(directory, onexc=ignore_missing_during_cleanup)
        except FileNotFoundError:
            pass
        db.delete_job(job_id)


def finalize_if_worker_absent(db: Database, settings: Settings, job_id: str) -> None:
    job = db.get_job(job_id)
    if job is None:
        return
    if job["status"] in TERMINAL_STATUSES:
        # An unrelated long document on this backend must not delay old-result cleanup.
        finalize_control(db, settings, job_id)
        return
    # Heartbeat age cannot prove that a process is dead. Use the same OS lock as
    # the worker; the inference child inherits it until all writes have stopped.
    with (settings.data_dir / f"worker-{job['backend']}.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        finalize_control(db, settings, job_id)
        db.heartbeat("offline", backend=job["backend"])
