import hashlib
import json
import os
import shutil
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Literal
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi import Path as PathParam
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from .backends import configurations
from .config import Settings
from .db import Database
from .exports import html_document, markdown_document
from .lifecycle import finalize_if_worker_absent
from .preview import render_preview
from .processing import document_path, validate_pdf
from .reports import html_report, processing_report
from .schemas import (
    TERMINAL_STATUSES,
    Comparison,
    DocumentResult,
    DocumentSelection,
    JobSummary,
    ReportSelection,
    RunCreate,
    RunFinish,
    WorkerStatus,
)

JobId = Annotated[str, PathParam(pattern=r"^[0-9a-f]{32}$")]


def worker_status(db: Database, settings: Settings, backend: str | None = None) -> WorkerStatus:
    record = db.worker(backend)
    if record is None:
        return WorkerStatus()
    status = WorkerStatus.model_validate(record)
    age = (datetime.now(UTC) - datetime.fromisoformat(status.heartbeat)).total_seconds()
    status.ready = status.state in {"idle", "processing"} and age < settings.worker_stale_seconds
    if age >= settings.worker_stale_seconds and status.state != "error":
        status.state = "offline"
    return status


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    db = Database(settings.db_path)

    @asynccontextmanager
    async def lifespan(_app):
        settings.setup()
        db.initialize()
        yield

    app = FastAPI(
        title="Document OCR",
        version="0.1.0",
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        description="Локальный CPU OCR: PDF → текст, таблицы, JSON, Markdown и HTML.",
    )
    app.state.db = db
    app.state.settings = settings

    @app.middleware("http")
    async def headers_and_limit(request, call_next):
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                oversized = int(content_length) > settings.max_upload_mb * 1024**2 + 1024**2
            except ValueError:
                return JSONResponse({"detail": "Некорректный Content-Length"}, status_code=400)
            if oversized:
                return JSONResponse({"detail": "Превышен лимит загрузки PDF"}, status_code=413)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self'; "
            "img-src 'self'; connect-src 'self'; frame-ancestors 'none'; object-src 'none'"
        )
        if request.url.path in {"/", "/docs"} or request.url.path.startswith("/static/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    def require_job(job_id: str) -> dict:
        job = db.get_job(job_id)
        if job is None or job["delete_requested"]:
            raise HTTPException(404, "Документ не найден")
        return job

    def result_for(job_id: str) -> DocumentResult:
        job = require_job(job_id)
        if job["status"] not in TERMINAL_STATUSES:
            raise HTTPException(409, "Обработка ещё не завершена", headers={"Retry-After": "2"})
        return DocumentResult(
            document=db.summary(job_id),
            engine=json.loads(job["engine_json"]),
            pages=db.pages(job_id),
        )

    @app.get("/health")
    def health():
        with db.connect() as connection:
            connection.execute("SELECT 1")
        workers = {backend: worker_status(db, settings, backend) for backend in settings.backends}
        return {"status": "ok", "worker": aggregate_status(workers), "workers": workers}

    def aggregate_status(workers: dict[str, WorkerStatus]) -> WorkerStatus:
        for state in ("processing", "idle", "error", "loading"):
            for status in workers.values():
                if status.state == state:
                    return status
        # Retain legacy heartbeat compatibility during upgrades.
        return worker_status(db, settings)

    @app.get("/ready")
    def ready():
        workers = {backend: worker_status(db, settings, backend) for backend in settings.backends}
        status = aggregate_status(workers)
        return JSONResponse(
            {
                "ready": status.ready,
                "worker": status.model_dump(),
                "workers": {key: value.model_dump() for key, value in workers.items()},
            },
            status_code=200 if status.ready else 503,
        )

    @app.get("/api/config")
    def public_config():
        return {
            "max_upload_mb": settings.max_upload_mb,
            "max_pages": settings.max_pages,
            "backends": configurations(settings),
        }

    def validate_backends(backends: list[str]):
        if (
            not backends
            or len(backends) != len(set(backends))
            or any(backend not in settings.backends for backend in backends)
        ):
            raise HTTPException(422, f"Выберите разные движки из: {', '.join(settings.backends)}")

    @app.post("/api/documents", response_model=JobSummary, status_code=202)
    async def upload_document(
        file: Annotated[UploadFile, File(description="PDF документ")],
        backend: Annotated[str, Form()] = "paddle",
        run_id: Annotated[str | None, Form(pattern=r"^[0-9a-f]{32}$")] = None,
        file_index: Annotated[int | None, Form(ge=0, le=999)] = None,
    ):
        validate_backends([backend])
        if (run_id is None) != (file_index is None):
            raise HTTPException(422, "Укажите запуск и позицию файла вместе")
        if run_id is not None:
            item = db.run_file(run_id, file_index)
            if item is None:
                raise HTTPException(404, "Файл запуска не найден")
            if item["backend"] != backend:
                raise HTTPException(409, "Провайдер не совпадает с провайдером запуска")
            if item["job_id"]:
                await file.close()
                require_job(item["job_id"])
                return db.summary(item["job_id"])
            if item["status"] == "accepted":
                raise HTTPException(410, "Документ запуска удалён")
        return await save_upload(file, backend, run_id=run_id, file_index=file_index)

    async def save_upload(
        file: UploadFile,
        backend: str,
        comparison_id: str | None = None,
        *,
        run_id: str | None = None,
        file_index: int | None = None,
    ):
        job_id = uuid4().hex
        directory = document_path(settings, job_id)
        directory.mkdir(parents=True)
        path = directory / "source.pdf"
        digest = hashlib.sha256()
        size = 0
        try:
            if file.size is not None and file.size > settings.max_upload_mb * 1024**2:
                raise HTTPException(413, "Превышен лимит загрузки PDF")
            with path.open("wb") as output:
                while chunk := await file.read(1024 * 1024):
                    size += len(chunk)
                    if size > settings.max_upload_mb * 1024**2:
                        raise HTTPException(413, "Превышен лимит загрузки PDF")
                    digest.update(chunk)
                    output.write(chunk)
            if size == 0:
                raise HTTPException(422, "Загружен пустой файл")
            try:
                pages = await run_in_threadpool(validate_pdf, path, settings.max_pages)
            except ValueError as error:
                raise HTTPException(422, str(error)) from error
            filename = (file.filename or "document.pdf").replace("\\", "/").rsplit("/", 1)[-1]
            filename = "".join(char for char in filename if char.isprintable())[:255]
            record = {
                "id": job_id,
                "filename": filename or "document.pdf",
                "digest": digest.hexdigest(),
                "pages": pages,
                "config": settings.recognition_config(),
                "backend": backend,
                "comparison_id": comparison_id,
                "run_id": run_id,
                "file_index": file_index,
            }
            if comparison_id is None:
                db.create_jobs([record])
        except BaseException:
            shutil.rmtree(directory, ignore_errors=True)
            raise
        finally:
            await file.close()
        return record if comparison_id else db.summary(job_id)

    @app.post("/api/comparisons", response_model=Comparison, status_code=202)
    async def compare_document(
        file: Annotated[UploadFile, File(description="Один PDF для всех конфигураций")],
        backends: Annotated[list[str], Form()],
    ):
        validate_backends(backends)
        comparison_id = uuid4().hex
        created = []
        try:
            first = await save_upload(file, backends[0], comparison_id)
            created.append(first["id"])
            records = [first]
            # Hard links avoid PDF copies; every job reads the same immutable source.
            for backend in backends[1:]:
                job_id = uuid4().hex
                created.append(job_id)
                directory = document_path(settings, job_id)
                directory.mkdir()
                os.link(
                    document_path(settings, first["id"]) / "source.pdf", directory / "source.pdf"
                )
                records.append({**first, "id": job_id, "backend": backend})
            db.create_jobs(records)
        except BaseException:
            for job_id in created:
                db.delete_job(job_id)
                shutil.rmtree(document_path(settings, job_id), ignore_errors=True)
            raise
        return Comparison(id=comparison_id, jobs=db.comparison_jobs(comparison_id))

    @app.get("/api/comparisons/{comparison_id}", response_model=Comparison)
    def get_comparison(comparison_id: JobId):
        jobs = db.comparison_jobs(comparison_id)
        if not jobs:
            raise HTTPException(404, "Сравнение не найдено")
        return Comparison(id=comparison_id, jobs=jobs)

    @app.get("/api/documents", response_model=list[JobSummary])
    def list_documents():
        return db.recent()

    def require_run(run_id: str) -> dict:
        run = db.run_details(run_id)
        if run is None:
            raise HTTPException(404, "Запуск не найден")
        return run

    @app.post("/api/runs", status_code=201)
    def create_run(selection: RunCreate):
        validate_backends([selection.backend])
        return require_run(db.create_run(selection.backend, selection.filenames))

    @app.get("/api/runs")
    def list_runs(
        limit: Annotated[int, Query(ge=1, le=100)] = 20,
        offset: Annotated[int, Query(ge=0)] = 0,
    ):
        return db.runs(limit, offset)

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: JobId):
        return require_run(run_id)

    @app.post("/api/runs/{run_id}/finish-upload")
    def finish_upload(run_id: JobId, selection: RunFinish):
        require_run(run_id)
        db.finish_run_upload(run_id, [item.model_dump() for item in selection.failures])
        return require_run(run_id)

    @app.post("/api/documents/status")
    def document_statuses(selection: DocumentSelection):
        documents, missing = [], []
        for job_id in dict.fromkeys(selection.document_ids):
            try:
                require_job(job_id)
                documents.append(db.summary(job_id))
            except (HTTPException, KeyError):
                missing.append(job_id)
        return {"documents": documents, "missing": missing}

    @app.post("/api/reports")
    def report(selection: ReportSelection, format: Literal["json", "html"] = "html"):
        return report_response(selection, format)

    def report_response(selection: ReportSelection, format: str, run: dict | None = None):
        if not selection.document_ids and not selection.upload_errors:
            raise HTTPException(422, "Выберите документы для отчёта")
        snapshots = []
        for job_id in dict.fromkeys(selection.document_ids):
            job = require_job(job_id)
            snapshots.append(
                DocumentResult(
                    document=db.summary(job_id),
                    engine=json.loads(job["engine_json"]),
                    pages=db.pages(job_id),
                )
            )
        value = processing_report(snapshots)
        value["upload_errors"] = [item.model_dump() for item in selection.upload_errors]
        value["totals"]["upload_errors"] = len(selection.upload_errors)
        if run is not None:
            value["run"] = {key: run[key] for key in ("id", "created_at", "backend")}
            value["complete"] = value["complete"] and not any(
                item["status"] == "pending" for item in run["files"]
            )
        if format == "html":
            content, media = html_report(value), "text/html"
        else:
            content, media = json.dumps(value, ensure_ascii=False, indent=2), "application/json"
        return Response(
            content,
            media_type=media,
            headers={
                "Content-Disposition": f'attachment; filename="ocr-processing-report.{format}"',
            },
        )

    @app.get("/api/runs/{run_id}/report")
    def run_report(run_id: JobId, format: Literal["json", "html"] = "html"):
        run = require_run(run_id)
        return report_response(
            ReportSelection(
                document_ids=[item["job_id"] for item in run["files"] if item["job"]],
                upload_errors=[
                    {
                        "filename": item["filename"],
                        "provider": run["backend"],
                        "error": item["error"] or "Файл ещё не загружен",
                    }
                    for item in run["files"]
                    if not item["job"]
                ],
            ),
            format,
            run,
        )

    @app.get("/api/documents/{job_id}", response_model=JobSummary)
    def get_document(job_id: JobId):
        require_job(job_id)
        return db.summary(job_id)

    @app.post("/api/documents/{job_id}/cancel", response_model=JobSummary, status_code=202)
    def cancel_document(job_id: JobId):
        require_job(job_id)
        db.request_cancel(job_id)
        db.finalize_unclaimed_cancel(job_id)
        finalize_if_worker_absent(db, settings, job_id)
        return db.summary(job_id)

    @app.delete("/api/documents/{job_id}")
    def delete_document(job_id: JobId):
        # Retried DELETE remains valid while asynchronous cleanup is pending.
        if db.get_job(job_id) is None:
            raise HTTPException(404, "Документ не найден")
        db.request_cancel(job_id, delete=True)
        db.finalize_unclaimed_cancel(job_id)
        finalize_if_worker_absent(db, settings, job_id)
        if db.get_job(job_id) is None:
            return JSONResponse({"id": job_id, "status": "deleted"})
        return JSONResponse({"id": job_id, "status": "deleting"}, status_code=202)

    @app.get("/api/documents/{job_id}/result", response_model=DocumentResult)
    def get_result(job_id: JobId):
        return result_for(job_id)

    @app.get("/api/documents/{job_id}/pages/{number}/preview")
    def page_preview(job_id: JobId, number: Annotated[int, PathParam(ge=1)]):
        result = result_for(job_id)
        page = next((page for page in result.pages if page.number == number), None)
        if page is None:
            raise HTTPException(404, "Страница не найдена")
        return {"html": render_preview(page.markdown or page.text)}

    @app.get("/api/documents/{job_id}/download")
    def download(job_id: JobId, format: Literal["json", "markdown", "html"] = "json"):
        result = result_for(job_id)
        if format == "markdown":
            content, media, suffix = markdown_document(result), "text/markdown", "md"
        elif format == "html":
            content, media, suffix = html_document(result), "text/html", "html"
        else:
            content, media, suffix = result.model_dump_json(indent=2), "application/json", "json"
        return Response(
            content,
            media_type=media,
            headers={"Content-Disposition": f'attachment; filename="document-{job_id}.{suffix}"'},
        )

    @app.get("/api/documents/{job_id}/pages/{number}/image")
    def page_image(job_id: JobId, number: Annotated[int, PathParam(ge=1)]):
        job = require_job(job_id)
        if number > job["page_count"]:
            raise HTTPException(404, "Страница не найдена")
        path = document_path(settings, job_id) / f"page-{number}.png"
        if not path.is_file():
            raise HTTPException(404, "Изображение страницы ещё не сохранено")
        return FileResponse(path, media_type="image/png")

    static = Path(__file__).with_name("static")
    app.mount("/static", StaticFiles(directory=static), name="static")

    def ui_page(filename: str) -> HTMLResponse:
        content = (static / filename).read_text(encoding="utf-8")
        for asset in ("app.js", "style.css"):
            version = hashlib.sha256((static / asset).read_bytes()).hexdigest()[:16]
            content = content.replace(f'"/static/{asset}"', f'"/static/{asset}?v={version}"')
        return HTMLResponse(content)

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def index():
        return ui_page("index.html")

    @app.get("/docs", response_class=HTMLResponse, include_in_schema=False)
    def api_docs():
        return ui_page("api.html")

    return app
