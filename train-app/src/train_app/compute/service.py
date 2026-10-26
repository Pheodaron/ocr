"""Отдельный HTTP-процесс для LightOn-4B или службы обучения."""

import fcntl
import hashlib
import json
import logging
import os
import subprocess
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from queue import Empty, Queue

from fastapi import FastAPI, HTTPException

from train_app.models import timestamp
from train_app.storage import Storage, atomic_json, read_json

logger = logging.getLogger(__name__)
ROLE = os.environ.get("TRAIN_COMPUTE_ROLE", "trainer")
STORAGE = Storage(Path(os.environ.get("TRAIN_DATA_DIR", "/data")))
ROOT = STORAGE.path(f"compute/{ROLE}")
ROOT.mkdir(parents=True, exist_ok=True)
CURRENT = {"state": "ready", "model_loaded": False}
PENDING: Queue[Path] = Queue()
SUBMISSIONS = threading.Lock()


def operation_dir(identifier: str) -> Path:
    if len(identifier) != 32 or any(char not in "0123456789abcdef" for char in identifier):
        raise HTTPException(422, "Недопустимый идентификатор")
    return ROOT / identifier


def status(path: Path, state: str, **values):
    atomic_json(path / "status.json", {"state": state, "updated_at": timestamp(), **values})


def worker(stop: threading.Event):
    native, gpu_lock = None, None
    while not stop.is_set():
        try:
            path = PENDING.get(timeout=0.5)
        except Empty:
            continue
        request_path = path / "request.json"
        current = read_json(path / "status.json")
        if current["state"] != "queued":
            continue
        payload = read_json(request_path)["payload"]

        def cancel(operation_path=path):
            return (operation_path / "cancel").exists() or stop.is_set()

        if cancel():
            status(path, "cancelled")
            continue
        CURRENT.update(state="running", operation_id=path.name)
        try:
            if ROLE == "lighton4b":
                if native is None:
                    gpu_lock = STORAGE.path("gpu.lock").open("a+")
                    while True:
                        try:
                            fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                            break
                        except BlockingIOError:
                            CURRENT["state"] = "waiting_gpu"
                            status(
                                path,
                                "waiting_gpu",
                                stage="Остановите другую службу, удерживающую GPU",
                            )
                            if cancel():
                                break
                            time.sleep(1)
                    if cancel():
                        status(path, "cancelled")
                        gpu_lock.close()
                        gpu_lock = None
                        continue
                    from .native import NativeModel

                    CURRENT["state"] = "loading"
                    status(path, "loading", stage="Загрузка LightOnOCR-3-4B")
                    native = NativeModel(
                        Path(os.environ.get("TRAIN_TEACHER_PATH", "/models/lighton4b"))
                    )
                    CURRENT["model_loaded"] = True
                CURRENT["state"] = "running"
                status(path, "running", stage="Распознавание страницы")
                output = native.recognize(
                    STORAGE.path(payload["image"]),
                    payload.get("prompt", "grounding"),
                    payload.get("longest_edge", 2048),
                    payload.get("max_output_tokens", 8192),
                    cancel,
                )
                atomic_json(path / "result.json", output)
                status(path, "cancelled" if cancel() else "completed")
            else:
                with (path / "log.txt").open("a") as log:
                    process = subprocess.Popen(
                        [sys.executable, "-m", "train_app.compute.runner", str(path)],
                        stdout=log,
                        stderr=subprocess.STDOUT,
                    )
                    status(path, "running", pid=process.pid)
                    while process.poll() is None:
                        if stop.wait(0.5):
                            (path / "cancel").touch()
                            process.terminate()
                            try:
                                process.wait(timeout=20)
                            except subprocess.TimeoutExpired:
                                process.kill()
                            break
                    current = read_json(path / "status.json")
                    if current["state"] not in {"completed", "cancelled", "failed"}:
                        status(
                            path,
                            "failed",
                            error=f"Вычислительный процесс завершён: {process.returncode}; смотрите журнал",
                        )
        except Exception as error:
            logger.exception("Operation %s failed", path.name)
            status(path, "failed", error=str(error))
            if native is None and gpu_lock:
                gpu_lock.close()
                gpu_lock = None
        finally:
            CURRENT.update(state="ready", operation_id=None)
    if gpu_lock:
        gpu_lock.close()


@asynccontextmanager
async def lifespan(_app):
    # Контейнер перезапущен: прежние процессы и их память отсутствуют.
    for path in ROOT.glob("*/status.json"):
        previous = read_json(path)
        if previous["state"] not in {"completed", "cancelled", "failed", "queued"}:
            status(
                path.parent,
                "failed",
                error="Служба перезапущена. Продолжите обучение из сохранённой контрольной точки",
            )
    for request in ROOT.glob("*/request.json"):
        if read_json(request.parent / "status.json")["state"] == "queued":
            PENDING.put(request.parent)
    stop = threading.Event()
    thread = threading.Thread(target=worker, args=(stop,), daemon=True)
    thread.start()
    yield
    stop.set()
    thread.join(timeout=25)


app = FastAPI(title=f"Вычислительная служба {ROLE}", lifespan=lifespan)


@app.get("/status")
def readiness():
    state = {"role": ROLE, **CURRENT}
    if identifier := CURRENT.get("operation_id"):
        path = operation_dir(identifier)
        if (path / "status.json").exists():
            state.update(read_json(path / "status.json"))
    return state


@app.put("/operations/{identifier}", status_code=202)
def submit(identifier: str, payload: dict):
    path = operation_dir(identifier)
    allowed = {"recognize"} if ROLE == "lighton4b" else {"train", "evaluate", "probe", "export"}
    if payload.get("kind") not in allowed:
        raise HTTPException(422, "Операция не поддерживается этой службой")
    fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    with SUBMISSIONS:
        if (path / "request.json").exists():
            if read_json(path / "request.json")["fingerprint"] != fingerprint:
                raise HTTPException(409, "Идентификатор уже связан с другими входными данными")
        else:
            path.mkdir(parents=True, exist_ok=True)
            # Сначала состояние, затем request: исполнитель видит только целую пару.
            status(path, "queued")
            atomic_json(path / "request.json", {"fingerprint": fingerprint, "payload": payload})
            PENDING.put(path)
    return {"id": identifier, "state": read_json(path / "status.json")["state"]}


@app.get("/operations/{identifier}")
def operation(identifier: str):
    path = operation_dir(identifier)
    if not (path / "status.json").exists():
        raise HTTPException(404)
    result = read_json(path / "status.json")
    if result["state"] == "completed":
        result["result"] = read_json(path / "result.json")
    if (path / "progress.json").exists():
        result["progress"] = read_json(path / "progress.json")
    return result


@app.post("/operations/{identifier}/cancel")
def cancel_operation(identifier: str):
    path = operation_dir(identifier)
    if not path.exists():
        raise HTTPException(404)
    (path / "cancel").touch()
    if read_json(path / "status.json")["state"] == "queued":
        status(path, "cancelled")
    return {"cancel_requested": True}


@app.get("/operations/{identifier}/log")
def operation_log(identifier: str):
    path = operation_dir(identifier) / "log.txt"
    if not path.exists():
        return {"text": ""}
    with path.open("rb") as stream:
        stream.seek(max(0, path.stat().st_size - 64_000))
        return {"text": stream.read().decode(errors="replace")}
