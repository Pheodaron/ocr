"""Устойчивая очередь: один исполнитель CPU и один диспетчер вычислительных служб."""

import asyncio
import fcntl
import logging
import time

import httpx
from sqlalchemy import select

from .datasets import training_history
from .documents import render_page
from .grounding import parse_grounding
from .metrics import compare
from .models import (
    Checkpoint,
    Comparison,
    Dataset,
    Document,
    Job,
    Page,
    Result,
    Task,
    timestamp,
)
from .services import qwen_idle, qwen_recognize, service_status
from .storage import atomic_json, read_json

logger = logging.getLogger(__name__)
TERMINAL = {"completed", "failed", "cancelled"}
COMPUTE_KINDS = {"lighton", "qwen", "train", "evaluate", "probe", "export"}


class Dispatcher:
    def __init__(self, database, storage, settings):
        self.database = database
        self.storage = storage
        self.settings = settings
        self.runners = []
        self.lock = None

    async def start(self):
        self.lock = self.storage.path("dispatcher.lock").open("a+")
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Диспетчер уже запущен; используйте один процесс API") from error
        self.runners = [asyncio.create_task(self.loop(False)), asyncio.create_task(self.loop(True))]

    async def stop(self):
        for runner in self.runners:
            runner.cancel()
        await asyncio.gather(*self.runners, return_exceptions=True)
        if self.lock:
            self.lock.close()

    def cancelled(self, job_id: str) -> bool:
        with self.database() as session:
            job = session.get(Job, job_id)
            return not job or job.cancel_requested

    def update(self, job_id: str, **values):
        with self.database() as session:
            job = session.get(Job, job_id)
            if job:
                for key, value in values.items():
                    setattr(job, key, value)
                job.updated_at = timestamp()
                session.commit()

    async def loop(self, compute: bool):
        async with httpx.AsyncClient(trust_env=False) as client:
            while True:
                try:
                    job = await self.pick_job(client, compute)
                    if job:
                        await self.execute(client, job)
                    else:
                        await asyncio.sleep(1)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.exception("Dispatcher iteration failed")
                    await asyncio.sleep(2)

    async def pick_job(self, client, compute):
        statuses = {}
        idle = {}
        with self.database() as session:
            jobs = session.scalars(
                select(Job).where(Job.status.not_in(TERMINAL)).order_by(Job.created_at)
            ).all()
        for job in jobs:
            if (job.kind in COMPUTE_KINDS) != compute:
                continue
            if job.cancel_requested:
                # Для удалённого задания сначала доставить отмену, а не только сменить надпись.
                if job.kind in COMPUTE_KINDS and job.kind != "qwen":
                    url = (
                        self.settings.lighton_url
                        if job.kind == "lighton"
                        else self.settings.trainer_url
                    )
                    with self.database() as session:
                        operations = session.scalars(
                            select(Task).where(Task.job_id == job.id, Task.status == "running")
                        ).all()
                    for task in operations:
                        try:
                            response = await client.post(
                                f"{url}/operations/{task.operation_id}/cancel", timeout=5
                            )
                            response.raise_for_status()
                        except httpx.HTTPError:
                            self.update(
                                job.id,
                                status="unknown",
                                error="Отмена ожидает восстановления связи со службой",
                            )
                            break
                    else:
                        self.finish_cancel(job.id)
                    if operations:
                        continue
                self.finish_cancel(job.id)
                continue
            if compute:
                name = (
                    "lighton4b"
                    if job.kind == "lighton"
                    else "qwen"
                    if job.kind == "qwen"
                    else "trainer"
                )
                url = {
                    "lighton4b": self.settings.lighton_url,
                    "qwen": self.settings.qwen_url,
                    "trainer": self.settings.trainer_url,
                }[name]
                if name not in statuses:
                    statuses[name] = await service_status(client, name, url)
                status = statuses[name]
                if not status["reachable"] or status["state"] in {
                    "loading",
                    "waiting_gpu",
                    "error",
                }:
                    self.update(
                        job.id,
                        status="waiting_service",
                        error=f"Ожидание службы {name}: {status['state']}",
                    )
                    continue
                if name == "qwen" and name not in idle:
                    idle[name] = await qwen_idle(client, url)
                if name == "qwen" and not idle[name]:
                    self.update(
                        job.id,
                        status="unknown",
                        error="Прежняя генерация Qwen ещё выполняется; ожидание освобождения слота",
                    )
                    continue
            return job
        return None

    def finish_cancel(self, job_id):
        with self.database() as session:
            job = session.get(Job, job_id)
            for task in session.scalars(
                select(Task).where(Task.job_id == job_id, Task.status.not_in(TERMINAL))
            ):
                task.status = "cancelled"
            job.status = "cancelled"
            session.commit()

    async def execute(self, client, job):
        self.update(job.id, status="running", error=None)
        with self.database() as session:
            task = session.scalars(
                select(Task)
                .where(Task.job_id == job.id, Task.status.not_in(TERMINAL))
                .order_by(Task.number)
            ).first()
            if not task:
                self.finalize(session, job.id)
                return
            recovered = task.status == "running"
            task.status = "running"
            if not recovered and job.kind != "qwen":
                task.attempt += 1
            session.commit()
        started = time.monotonic()
        try:
            if job.kind == "ingest":
                await asyncio.to_thread(self.ingest, job, task)
            elif job.kind == "compare":
                await asyncio.to_thread(self.compare_page, task)
            elif job.kind in {"lighton", "qwen"}:
                await self.recognize(client, job, task)
            else:
                await self.compute(client, job, task)
            with self.database() as session:
                stored = session.get(Task, task.id)
                stored.status = "cancelled" if self.cancelled(job.id) else "completed"
                stored.error = None
                session.commit()
        except asyncio.CancelledError:
            # При остановке app удалённое задание сохраняет свой идентификатор для сверки.
            if self.cancelled(job.id):
                self.finish_cancel(job.id)
                return
            raise
        except (httpx.TransportError, httpx.TimeoutException) as error:
            self.update(job.id, status="unknown", error=f"Связь со службой потеряна: {error}")
            await asyncio.sleep(2)
            return
        except Exception as error:
            logger.exception("Job %s task %s failed", job.id, task.id)
            with self.database() as session:
                stored = session.get(Task, task.id)
                stored.status, stored.error = "failed", str(error)
                if job.kind in {"lighton", "qwen"} and not session.scalar(
                    select(Result).where(Result.operation_id == task.operation_id)
                ):
                    session.add(
                        Result(
                            page_id=task.page_id,
                            engine=job.kind,
                            operation_id=task.operation_id,
                            status="failed",
                            error=str(error),
                            raw_path=None,
                            seconds=time.monotonic() - started,
                        )
                    )
                session.commit()
        logger.info("Job %s task %s finished in %.2fs", job.id, task.id, time.monotonic() - started)
        with self.database() as session:
            self.finalize(session, job.id)

    def finalize(self, session, job_id):
        job = session.get(Job, job_id)
        tasks = session.scalars(select(Task).where(Task.job_id == job_id)).all()
        job.completed = sum(task.status == "completed" for task in tasks)
        job.failed = sum(task.status == "failed" for task in tasks)
        job.updated_at = timestamp()
        if all(task.status in TERMINAL for task in tasks):
            job.status = (
                "cancelled"
                if job.cancel_requested
                else "failed"
                if job.failed == len(tasks) and tasks
                else "completed"
            )
        else:
            job.status = "queued"
        if job.kind == "ingest" and job.status in TERMINAL:
            document = session.get(Document, job.config["document_id"])
            document.status = "ready" if job.completed else "failed"
            document.error = f"Ошибочных страниц: {job.failed}" if job.failed else None
        session.commit()

    def ingest(self, job, task):
        with self.database() as session:
            document = session.get(Document, job.config["document_id"])
            existing = session.scalar(
                select(Page).where(Page.document_id == document.id, Page.number == task.number)
            )
            if existing:
                return
            page = render_page(document, task.number, self.storage, self.settings)
            session.add(page)
            session.commit()

    def compare_page(self, task):
        with self.database() as session:
            left = session.scalars(
                select(Result)
                .where(
                    Result.page_id == task.page_id,
                    Result.engine == "lighton",
                    Result.status == "completed",
                )
                .order_by(Result.created_at.desc())
            ).first()
            right = session.scalars(
                select(Result)
                .where(
                    Result.page_id == task.page_id,
                    Result.engine == "qwen",
                    Result.status == "completed",
                )
                .order_by(Result.created_at.desc())
            ).first()
            if not left or not right:
                raise ValueError("Для сравнения нужны успешные результаты обеих моделей")
            details = compare(left.markdown, right.markdown)
            session.add(
                Comparison(
                    page_id=task.page_id,
                    lighton_result_id=left.id,
                    qwen_result_id=right.id,
                    agrees=details["agrees"],
                    details=details,
                )
            )
            session.commit()

    async def remote_operation(self, client, url, operation_id, payload, job_id):
        request_path = (
            self.storage.project_dir(payload["project_id"])
            / "runs"
            / job_id
            / f"{operation_id}.request.json"
        )
        if request_path.exists():
            payload = read_json(request_path)
        else:
            atomic_json(request_path, payload)
        response = await client.put(f"{url}/operations/{operation_id}", json=payload, timeout=15)
        response.raise_for_status()
        while True:
            if self.cancelled(job_id):
                response = await client.post(f"{url}/operations/{operation_id}/cancel", timeout=10)
                response.raise_for_status()
            response = await client.get(f"{url}/operations/{operation_id}", timeout=10)
            response.raise_for_status()
            state = response.json()
            self.update(job_id, error=None, status=state.get("state", "running"))
            if state.get("progress"):
                atomic_json(
                    self.storage.path(f"operations/{job_id}.progress.json"), state["progress"]
                )
            if state["state"] == "completed":
                return state["result"]
            if state["state"] == "cancelled":
                raise asyncio.CancelledError()
            if state["state"] == "failed":
                raise RuntimeError(state.get("error", "Ошибка вычислительной службы"))
            await asyncio.sleep(1)

    async def recognize(self, client, job, task):
        with self.database() as session:
            page = session.get(Page, task.page_id)
            if session.scalar(select(Result).where(Result.operation_id == task.operation_id)):
                return
            project = session.get(Document, page.document_id).project_id
        started = time.monotonic()
        if job.kind == "qwen":
            if task.attempt >= 2:
                raise RuntimeError("Достигнут лимит повторов потерянной генерации Qwen")
            with self.database() as session:
                stored = session.get(Task, task.id)
                stored.attempt += 1
                session.commit()
            generation = asyncio.create_task(
                qwen_recognize(
                    client,
                    self.settings,
                    self.storage.path(page.image_path),
                    lambda: self.cancelled(job.id),
                    lambda value: atomic_json(
                        self.storage.path(f"operations/{job.id}.progress.json"), value
                    ),
                )
            )
            try:
                while not generation.done():
                    if self.cancelled(job.id):
                        generation.cancel()
                        raise asyncio.CancelledError()
                    await asyncio.wait({generation}, timeout=1)
                output = await generation
            finally:
                if not generation.done():
                    generation.cancel()
                    await asyncio.gather(generation, return_exceptions=True)
        else:
            output = await self.remote_operation(
                client,
                self.settings.lighton_url,
                task.operation_id,
                {
                    "kind": "recognize",
                    "project_id": project,
                    "image": page.image_path,
                    "prompt": "grounding",
                    "longest_edge": self.settings.model_longest_edge,
                    "max_output_tokens": self.settings.max_output_tokens,
                },
                job.id,
            )
        raw_path = self.storage.project_dir(project) / "raw" / f"{task.operation_id}.json"
        atomic_json(raw_path, output)
        markdown, blocks, warnings = parse_grounding(output.get("text", ""))
        error = output.get("error")
        with self.database() as session:
            if self.cancelled(job.id):
                return
            session.add(
                Result(
                    page_id=page.id,
                    engine=job.kind,
                    operation_id=task.operation_id,
                    status="failed" if error else "completed",
                    markdown=markdown,
                    blocks=blocks,
                    raw_path=self.storage.relative(raw_path),
                    error=error,
                    metadata_json={**output.get("metadata", {}), "warnings": warnings},
                    seconds=time.monotonic() - started,
                )
            )
            session.commit()
        if error:
            raise RuntimeError(error)

    async def compute(self, client, job, task):
        payload = {
            **job.config,
            "kind": job.kind,
            "project_id": job.project_id,
            "output_dir": f"projects/{job.project_id}/runs/{job.id}",
        }
        with self.database() as session:
            if dataset_id := job.config.get("dataset_id"):
                dataset = session.get(Dataset, dataset_id)
                payload["manifest"] = dataset.manifest_path
            payload["training_history"] = training_history(session, self.storage, job.project_id)
            if checkpoint_id := job.config.get("checkpoint_id") or job.config.get(
                "resume_checkpoint_id"
            ):
                checkpoint = session.get(Checkpoint, checkpoint_id)
                if not checkpoint or checkpoint.project_id != job.project_id:
                    raise ValueError("Контрольная точка не принадлежит проекту")
                payload["checkpoint"] = checkpoint.path
            if job.kind == "probe":
                payload["image"] = session.get(Page, task.page_id).image_path
            if job.kind == "train":
                stored = session.get(Job, job.id)
                stored.config = {**stored.config, "training_started": True}
                session.commit()
        result = await self.remote_operation(
            client, self.settings.trainer_url, task.operation_id, payload, job.id
        )
        if job.kind == "probe":
            with self.database() as session:
                page = session.get(Page, task.page_id)
                markdown, blocks, warnings = parse_grounding(result.get("text", ""))
                result = {
                    **result,
                    "markdown": markdown,
                    "blocks": blocks,
                    "page": {"id": page.id, "width": page.width, "height": page.height},
                    "warnings": warnings,
                }
        path = (
            self.storage.project_dir(job.project_id)
            / "runs"
            / job.id
            / f"{task.operation_id}.result.json"
        )
        atomic_json(path, result)
        with self.database() as session:
            stored = session.get(Job, job.id)
            stored.result_path = self.storage.relative(path)
            for item in result.get("checkpoints", []):
                if not session.scalar(select(Checkpoint).where(Checkpoint.path == item["path"])):
                    session.add(
                        Checkpoint(
                            project_id=job.project_id,
                            job_id=job.id,
                            path=item["path"],
                            step=item["step"],
                            metadata_json=item,
                        )
                    )
            session.commit()
