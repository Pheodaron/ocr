import fcntl
import logging
import os
import signal
import sys
import time
from pathlib import Path

from train_app.storage import Storage, atomic_json, read_json


def main(path: Path):
    logging.basicConfig(level=logging.INFO)
    storage = Storage(Path(os.environ.get("TRAIN_DATA_DIR", "/data")))
    payload = read_json(path / "request.json")["payload"]

    def cancelled():
        return (path / "cancel").exists()

    def request_stop(_signal, _frame):
        (path / "cancel").touch()

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    def state(value, **extra):
        atomic_json(path / "status.json", {"state": value, **extra})

    def progress(value):
        atomic_json(path / "progress.json", value)

    with storage.path("gpu.lock").open("a+") as lock:
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                state("waiting_gpu", stage="Остановите другую службу, удерживающую GPU")
                if cancelled():
                    state("cancelled")
                    return
                time.sleep(1)
        try:
            state("running")
            if payload["kind"] == "train":
                from .training import train

                result = train(payload, storage, cancelled, progress)
            elif payload["kind"] == "evaluate":
                from .evaluation import evaluate

                result = evaluate(payload, storage, cancelled, progress)
            elif payload["kind"] == "probe":
                from .native import NativeModel

                checkpoint = (
                    storage.path(payload["checkpoint"]) if payload.get("checkpoint") else None
                )
                native = NativeModel(
                    Path(os.environ.get("TRAIN_STUDENT_PATH", "/models/lighton08")), checkpoint
                )
                result = native.recognize(
                    storage.path(payload["image"]),
                    payload.get("prompt", ""),
                    payload.get("longest_edge", 2048),
                    payload.get("max_output_tokens", 8192),
                    cancelled,
                )
            elif payload["kind"] == "export":
                from .exporting import export_model

                result = export_model(payload, storage, cancelled, progress)
            else:
                raise ValueError("Неизвестная операция")
            atomic_json(path / "result.json", result)
            # При штатной остановке обучения результат с контрольными точками тоже доступен.
            state("completed", cancelled=cancelled())
        except Exception as error:
            logging.exception("Operation failed")
            state("failed", error=str(error))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
