from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy import inspect, select
from sqlalchemy.orm import Session

from . import schemas
from .annotations import save_annotation
from .datasets import assign_splits, create_dataset, export_dataset, with_validation
from .documents import duplicate_candidates, inspect_document
from .models import (
    Annotation,
    Checkpoint,
    Comparison,
    Dataset,
    Document,
    Job,
    Page,
    Project,
    Result,
    Task,
    new_id,
)
from .services import all_statuses
from .storage import digest, read_json

router = APIRouter(prefix="/api")


def database(request: Request):
    with request.app.state.database() as session:
        yield session


DB = Depends(database)


def record(value):
    return {column.key: getattr(value, column.key) for column in inspect(value).mapper.column_attrs}


def get_project(session, project_id: str, writable=False):
    project = session.get(Project, project_id)
    if not project:
        raise HTTPException(404, "Проект не найден")
    if writable and project.archived:
        raise HTTPException(409, "Проект архивирован")
    return project


def get_document(session, project_id, document_id):
    document = session.get(Document, document_id)
    if not document or document.project_id != project_id:
        raise HTTPException(404, "Документ не найден в проекте")
    return document


def get_page(session, project_id, page_id):
    page = session.get(Page, page_id)
    if not page:
        raise HTTPException(404, "Страница не найдена")
    get_document(session, project_id, page.document_id)
    return page


@router.get("/projects")
def projects(session: Session = DB):
    return [
        record(project)
        for project in session.scalars(select(Project).order_by(Project.created_at.desc()))
    ]


@router.post("/projects", status_code=201)
def add_project(payload: schemas.ProjectInput, session: Session = DB):
    project = Project(**payload.model_dump())
    session.add(project)
    session.commit()
    return record(project)


@router.patch("/projects/{project_id}")
def edit_project(project_id: str, payload: schemas.ProjectInput, session: Session = DB):
    project = get_project(session, project_id)
    for key, value in payload.model_dump().items():
        setattr(project, key, value)
    session.commit()
    return record(project)


@router.get("/services")
async def services(request: Request):
    return await all_statuses(request.app.state.settings)


@router.get("/projects/{project_id}/documents")
def documents(project_id: str, session: Session = DB):
    get_project(session, project_id)
    items = []
    for document in session.scalars(
        select(Document).where(Document.project_id == project_id).order_by(Document.created_at)
    ):
        pages = session.scalars(select(Page).where(Page.document_id == document.id)).all()
        annotations = [
            session.get(Annotation, page.current_annotation_id)
            for page in pages
            if page.current_annotation_id
        ]
        items.append(
            {
                **record(document),
                "rendered_pages": len(pages),
                "approved_pages": sum(a.text_approved for a in annotations),
                "geometry_pages": sum(a.geometry_approved for a in annotations),
            }
        )
    return items


@router.post("/projects/{project_id}/documents", status_code=202)
async def upload_document(
    project_id: str, file: UploadFile, request: Request, session: Session = DB
):
    get_project(session, project_id, True)
    storage, settings = request.app.state.storage, request.app.state.settings
    identifier = new_id()
    filename = Path(file.filename or "document").name
    suffix = Path(filename).suffix.lower()
    if suffix not in {".pdf", ".png", ".jpg", ".jpeg"}:
        raise HTTPException(415, "Поддерживаются PDF, PNG и JPEG")
    path = storage.project_dir(project_id) / "sources" / f"{identifier}{suffix}"
    path.parent.mkdir(parents=True, exist_ok=True)
    size = 0
    try:
        with path.open("wb") as output:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > settings.max_upload_mb * 1024 * 1024:
                    raise HTTPException(413, f"Лимит файла: {settings.max_upload_mb} МБ")
                output.write(chunk)
        import asyncio

        count = await asyncio.to_thread(inspect_document, path, suffix, settings.max_document_pages)
        checksum = await asyncio.to_thread(digest, path)
        document = Document(
            id=identifier,
            project_id=project_id,
            filename=filename,
            source_path=storage.relative(path),
            sha256=checksum,
            page_count=count,
        )
        session.add(document)
        session.flush()
        job = Job(
            project_id=project_id, kind="ingest", config={"document_id": identifier}, total=count
        )
        session.add(job)
        session.flush()
        for number in range(1, count + 1):
            session.add(Task(job_id=job.id, number=number))
        session.commit()
        return record(document)
    except Exception as error:
        path.unlink(missing_ok=True)
        if isinstance(error, HTTPException):
            raise
        raise HTTPException(422, str(error)) from error
    finally:
        await file.close()


@router.get("/projects/{project_id}/documents/{document_id}/pages")
def pages(project_id: str, document_id: str, session: Session = DB):
    get_document(session, project_id, document_id)
    return [
        record(page)
        for page in session.scalars(
            select(Page).where(Page.document_id == document_id).order_by(Page.number)
        )
    ]


@router.get("/projects/{project_id}/pages/{page_id}")
def page_details(project_id: str, page_id: str, session: Session = DB):
    page = get_page(session, project_id, page_id)
    annotation = (
        session.get(Annotation, page.current_annotation_id) if page.current_annotation_id else None
    )
    return {
        "page": record(page),
        "document": record(session.get(Document, page.document_id)),
        "annotation": record(annotation) if annotation else None,
        "results": [
            record(item)
            for item in session.scalars(
                select(Result).where(Result.page_id == page_id).order_by(Result.created_at.desc())
            )
        ],
        "comparisons": [
            record(item)
            for item in session.scalars(
                select(Comparison)
                .where(Comparison.page_id == page_id)
                .order_by(Comparison.created_at.desc())
            )
        ],
        "history": [
            record(item)
            for item in session.scalars(
                select(Annotation)
                .where(Annotation.page_id == page_id)
                .order_by(Annotation.created_at.desc())
            )
        ],
    }


@router.get("/projects/{project_id}/pages/{page_id}/image")
def page_image(project_id: str, page_id: str, request: Request, session: Session = DB):
    page = get_page(session, project_id, page_id)
    return FileResponse(request.app.state.storage.path(page.image_path), media_type="image/png")


@router.get("/projects/{project_id}/results/{result_id}/raw")
def raw_result(project_id: str, result_id: str, request: Request, session: Session = DB):
    result = session.get(Result, result_id)
    if not result:
        raise HTTPException(404)
    get_page(session, project_id, result.page_id)
    if not result.raw_path:
        raise HTTPException(404)
    return FileResponse(
        request.app.state.storage.path(result.raw_path),
        media_type="application/json",
        filename=f"result-{result.id}.json",
    )


@router.post("/projects/{project_id}/pages/{page_id}/annotations", status_code=201)
def annotate(
    project_id: str, page_id: str, payload: schemas.AnnotationInput, session: Session = DB
):
    get_project(session, project_id, True)
    page = get_page(session, project_id, page_id)
    if payload.source_result_id:
        source = session.get(Result, payload.source_result_id)
        if not source or source.page_id != page.id:
            raise HTTPException(422, "Ответ не принадлежит странице")
    try:
        annotation = save_annotation(session, page, payload)
        session.commit()
        return record(annotation)
    except ValueError as error:
        raise HTTPException(409, str(error)) from error


@router.patch("/projects/{project_id}/splits")
def splits(project_id: str, payload: schemas.SplitInput, session: Session = DB):
    get_project(session, project_id, True)
    for identifier in payload.document_ids:
        document = get_document(session, project_id, identifier)
        if payload.split is not None:
            document.split = payload.split
        if payload.group is not None:
            document.group = payload.group
    session.commit()
    return {"updated": len(payload.document_ids)}


@router.post("/projects/{project_id}/splits/automatic")
def automatic_splits(project_id: str, payload: schemas.AutoSplitInput, session: Session = DB):
    get_project(session, project_id, True)
    try:
        result = assign_splits(session, project_id, payload.fractions, payload.seed)
        session.commit()
        return result
    except ValueError as error:
        raise HTTPException(422, str(error)) from error


@router.get("/projects/{project_id}/duplicates")
def duplicates(project_id: str, session: Session = DB):
    get_project(session, project_id)
    return duplicate_candidates(session, project_id)


@router.get("/projects/{project_id}/datasets")
def datasets(project_id: str, session: Session = DB):
    get_project(session, project_id)
    return [
        record(item)
        for item in session.scalars(
            select(Dataset)
            .where(Dataset.project_id == project_id)
            .order_by(Dataset.created_at.desc())
        )
    ]


@router.post("/projects/{project_id}/datasets", status_code=201)
def snapshot(
    project_id: str, payload: schemas.DatasetInput, request: Request, session: Session = DB
):
    get_project(session, project_id, True)
    try:
        dataset = create_dataset(session, request.app.state.storage, project_id, payload.name)
        session.commit()
        return record(dataset)
    except ValueError as error:
        raise HTTPException(422, str(error)) from error


@router.get("/projects/{project_id}/datasets/{dataset_id}")
def dataset_details(project_id: str, dataset_id: str, request: Request, session: Session = DB):
    dataset = session.get(Dataset, dataset_id)
    if not dataset or dataset.project_id != project_id:
        raise HTTPException(404)
    return read_json(request.app.state.storage.path(dataset.manifest_path))


@router.get("/projects/{project_id}/datasets/{dataset_id}/download")
def dataset_download(project_id: str, dataset_id: str, request: Request, session: Session = DB):
    dataset = session.get(Dataset, dataset_id)
    if not dataset or dataset.project_id != project_id:
        raise HTTPException(404)
    path = export_dataset(request.app.state.storage, dataset)
    return FileResponse(path, media_type="application/zip", filename=f"dataset-{dataset_id}.zip")


@router.get("/projects/{project_id}/jobs")
def jobs(project_id: str, session: Session = DB):
    get_project(session, project_id)
    return [
        record(item)
        for item in session.scalars(
            select(Job)
            .where(Job.project_id == project_id, Job.hidden.is_(False))
            .order_by(Job.created_at.desc())
        )
    ]


@router.post("/projects/{project_id}/jobs", status_code=202)
def add_job(project_id: str, payload: schemas.JobInput, request: Request, session: Session = DB):
    get_project(session, project_id, True)
    config = dict(payload.config)
    if payload.kind == "train":
        try:
            config = schemas.TrainingConfig.model_validate(config).model_dump()
        except ValueError as error:
            raise HTTPException(422, str(error)) from error
    elif payload.kind in {"evaluate", "probe", "export"}:
        try:
            config = schemas.InferenceConfig.model_validate(config).model_dump()
        except ValueError as error:
            raise HTTPException(422, str(error)) from error
    if config.get("dataset_id"):
        selected_dataset = session.get(Dataset, config["dataset_id"])
        if not selected_dataset or selected_dataset.project_id != project_id:
            raise HTTPException(422, "Набор данных не принадлежит проекту")
    if payload.kind in {"train", "evaluate"}:
        dataset = session.get(Dataset, config.get("dataset_id"))
        if not dataset or dataset.project_id != project_id:
            raise HTTPException(422, "Выберите версию данных проекта")
        if payload.kind == "train":
            try:
                dataset = with_validation(session, request.app.state.storage, dataset)
                config["dataset_id"] = dataset.id
            except ValueError as error:
                raise HTTPException(422, str(error)) from error
        if payload.kind == "evaluate" and not dataset.counts["test"]["pages"]:
            raise HTTPException(422, "В версии данных нет тестовой выборки")
    if payload.kind == "export" and not config.get("checkpoint_id"):
        raise HTTPException(422, "Для экспорта выберите контрольную точку")
    for key in ("checkpoint_id", "resume_checkpoint_id"):
        if config.get(key):
            checkpoint = session.get(Checkpoint, config[key])
            if not checkpoint or checkpoint.project_id != project_id:
                raise HTTPException(422, "Контрольная точка не принадлежит проекту")
            if payload.kind == "export" and not config.get("dataset_id"):
                config["dataset_id"] = session.get(Job, checkpoint.job_id).config.get("dataset_id")
    job = Job(project_id=project_id, kind=payload.kind, config=config)
    session.add(job)
    session.flush()
    if payload.kind in {"train", "evaluate", "export"}:
        session.add(Task(job_id=job.id))
        job.total = 1
    else:
        statement = select(Page).join(Document).where(Document.project_id == project_id)
        if payload.document_ids:
            statement = statement.where(Document.id.in_(payload.document_ids))
        if payload.page_ids:
            statement = statement.where(Page.id.in_(payload.page_ids))
        pages = session.scalars(statement.order_by(Document.created_at, Page.number)).all()
        for page in pages:
            previous = session.scalars(
                select(Result)
                .where(Result.page_id == page.id, Result.engine == payload.kind)
                .order_by(Result.created_at.desc())
            ).first()
            if payload.kind in {"lighton", "qwen"}:
                if payload.mode == "missing" and previous and previous.status == "completed":
                    continue
                if payload.mode == "failed" and (not previous or previous.status != "failed"):
                    continue
            session.add(Task(job_id=job.id, page_id=page.id, number=job.total + 1))
            job.total += 1
    if not job.total:
        raise HTTPException(
            422, "Нет подходящих страниц. Проверьте подготовку документов и режим запуска"
        )
    session.commit()
    return record(job)


@router.get("/projects/{project_id}/jobs/{job_id}")
def job_details(project_id: str, job_id: str, request: Request, session: Session = DB):
    job = session.get(Job, job_id)
    if not job or job.project_id != project_id:
        raise HTTPException(404)
    result = read_json(request.app.state.storage.path(job.result_path)) if job.result_path else None
    progress_path = request.app.state.storage.path(f"operations/{job_id}.progress.json")
    history_path = request.app.state.storage.path(
        f"projects/{project_id}/runs/{job_id}/history.json"
    )
    return {
        **record(job),
        "tasks": [
            record(task)
            for task in session.scalars(
                select(Task).where(Task.job_id == job_id).order_by(Task.number)
            )
        ],
        "result": result,
        "progress": read_json(progress_path) if progress_path.exists() else None,
        "training_history": read_json(history_path)[-2000:] if history_path.exists() else [],
    }


@router.get("/projects/{project_id}/jobs/{job_id}/log")
async def job_log(project_id: str, job_id: str, request: Request, session: Session = DB):
    import httpx

    job = session.get(Job, job_id)
    if not job or job.project_id != project_id:
        raise HTTPException(404)
    if job.kind in {"qwen", "ingest", "compare"}:
        return {"text": "Журнал доступен через podman compose logs соответствующей службы."}
    task = session.scalars(select(Task).where(Task.job_id == job_id)).first()
    if not task:
        return {"text": ""}
    settings = request.app.state.settings
    url = settings.lighton_url if job.kind == "lighton" else settings.trainer_url
    try:
        async with httpx.AsyncClient(trust_env=False) as client:
            response = await client.get(f"{url}/operations/{task.operation_id}/log", timeout=3)
            response.raise_for_status()
            return response.json()
    except httpx.HTTPError:
        return {"text": "Служба недоступна; сохранённые журналы находятся в data/compute/."}


@router.post("/projects/{project_id}/jobs/{job_id}/cancel")
def cancel_job(project_id: str, job_id: str, session: Session = DB):
    job = session.get(Job, job_id)
    if not job or job.project_id != project_id:
        raise HTTPException(404)
    job.cancel_requested = True
    session.commit()
    return record(job)


@router.delete("/projects/{project_id}/jobs/{job_id}")
def delete_job(project_id: str, job_id: str, session: Session = DB):
    job = session.get(Job, job_id)
    if not job or job.project_id != project_id:
        raise HTTPException(404)
    if job.status not in {"completed", "failed", "cancelled"}:
        raise HTTPException(409, "Сначала остановите задание")
    if job.kind == "train" and job.config.get("training_started"):
        job.hidden = True
        session.commit()
        return {"deleted": True, "provenance_retained": True}
    for task in session.scalars(select(Task).where(Task.job_id == job_id)):
        session.delete(task)
    session.delete(job)
    session.commit()
    return {"deleted": True}


@router.get("/projects/{project_id}/checkpoints")
def checkpoints(project_id: str, request: Request, session: Session = DB):
    get_project(session, project_id)
    storage = request.app.state.storage
    for marker in storage.project_dir(project_id).glob("runs/*/checkpoint-*/complete.json"):
        metadata = read_json(marker)
        job_id = marker.parent.parent.name
        job = session.get(Job, job_id)
        if not job or job.project_id != project_id or job.kind != "train":
            continue
        path = storage.relative(marker.parent)
        if not session.scalar(select(Checkpoint).where(Checkpoint.path == path)):
            session.add(
                Checkpoint(
                    project_id=project_id,
                    job_id=job_id,
                    path=path,
                    step=metadata["step"],
                    metadata_json=metadata,
                )
            )
    session.commit()
    return [
        record(item)
        for item in session.scalars(
            select(Checkpoint)
            .where(Checkpoint.project_id == project_id)
            .order_by(Checkpoint.created_at.desc())
        )
    ]


@router.get("/projects/{project_id}/backup")
def backup(project_id: str, request: Request, session: Session = DB):
    from .backup import export_project

    get_project(session, project_id)
    try:
        path = export_project(session, request.app.state.storage, project_id)
        return FileResponse(
            path, media_type="application/zip", filename=f"project-{project_id}.zip"
        )
    except ValueError as error:
        raise HTTPException(409, str(error)) from error


@router.post("/restore", status_code=201)
async def restore(file: UploadFile, request: Request, session: Session = DB):
    from zipfile import BadZipFile

    from sqlalchemy.exc import IntegrityError

    from .backup import restore_project

    storage = request.app.state.storage
    path = storage.path(f"restore-{new_id()}.zip")
    try:
        size = 0
        with path.open("wb") as target:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > 50 * 1024**3:
                    raise HTTPException(413, "Резервная копия превышает 50 ГБ")
                target.write(chunk)
        import asyncio

        identifier = await asyncio.to_thread(restore_project, session, storage, path)
        return {"project_id": identifier}
    except (ValueError, KeyError, BadZipFile, IntegrityError) as error:
        raise HTTPException(422, str(error)) from error
    finally:
        path.unlink(missing_ok=True)
        await file.close()


@router.delete("/projects/{project_id}")
def remove_project(project_id: str, request: Request, session: Session = DB):
    from .backup import delete_project

    get_project(session, project_id)
    try:
        delete_project(session, request.app.state.storage, project_id)
        return {"deleted": True}
    except ValueError as error:
        raise HTTPException(409, str(error)) from error
