"""Проверяемая резервная копия одного проекта; восстановление без слияния версий."""

import json
import shutil
import sqlite3
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

from sqlalchemy import select

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
from .storage import digest

TABLES = [Project, Document, Page, Result, Comparison, Annotation, Dataset, Job, Task, Checkpoint]


def export_project(session, storage, project_id):
    with storage.project_lock(project_id, shared=True):
        project = session.get(Project, project_id)
        if not project:
            raise ValueError("Проект не найден")
        active = session.scalar(
            select(Job).where(
                Job.project_id == project_id,
                Job.status.not_in({"completed", "failed", "cancelled"}),
            )
        )
        if active:
            raise ValueError("Перед резервным копированием завершите или отмените задания проекта")
        destination = storage.path(f"exports/project-{project_id}-{new_id()}.zip")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=storage.root) as temporary:
            snapshot = Path(temporary) / "snapshot.sqlite"
            # Штатный backup учитывает WAL и создаёт согласованную читаемую базу.
            with (
                sqlite3.connect(storage.root / "workbench.sqlite") as source,
                sqlite3.connect(snapshot) as target,
            ):
                source.backup(target)
            with sqlite3.connect(snapshot) as connection:
                connection.row_factory = sqlite3.Row
                document_ids = [
                    row[0]
                    for row in connection.execute(
                        "SELECT id FROM documents WHERE project_id=?", (project_id,)
                    )
                ]
                page_ids = [
                    row[0]
                    for identifier in document_ids
                    for row in connection.execute(
                        "SELECT id FROM pages WHERE document_id=?", (identifier,)
                    )
                ]
                job_ids = [
                    row[0]
                    for row in connection.execute(
                        "SELECT id FROM jobs WHERE project_id=?", (project_id,)
                    )
                ]
                records = {}
                for model in TABLES:
                    column = (
                        "project_id"
                        if hasattr(model, "project_id")
                        else "document_id"
                        if model is Page
                        else "job_id"
                        if model is Task
                        else "page_id"
                        if hasattr(model, "page_id")
                        else "id"
                    )
                    identifiers = (
                        document_ids
                        if model is Page
                        else job_ids
                        if model is Task
                        else page_ids
                        if hasattr(model, "page_id")
                        else [project_id]
                    )
                    rows = []
                    for identifier in identifiers:
                        rows.extend(
                            dict(row)
                            for row in connection.execute(
                                f'SELECT * FROM "{model.__tablename__}" WHERE "{column}"=?',
                                (identifier,),
                            )
                        )
                    records[model.__tablename__] = rows
            files = {}
            project_root = storage.project_dir(project_id)
            with zipfile.ZipFile(
                destination.with_suffix(".tmp"), "w", zipfile.ZIP_DEFLATED
            ) as archive:
                for path in project_root.rglob("*"):
                    if path.is_file() and not path.is_symlink():
                        relative = storage.relative(path)
                        files[relative] = {"sha256": digest(path), "size": path.stat().st_size}
                        archive.write(path, relative)
                archive.writestr(
                    "backup.json",
                    json.dumps(
                        {
                            "schema_version": 1,
                            "project_id": project_id,
                            "records": records,
                            "files": files,
                        },
                        ensure_ascii=False,
                    ),
                )
            destination.with_suffix(".tmp").replace(destination)
        return destination


def restore_project(session, storage, source: Path):
    with zipfile.ZipFile(source) as archive:
        if sum(item.file_size for item in archive.infolist()) > 100 * 1024**3:
            raise ValueError("Распакованный архив превышает 100 ГБ")
        if archive.getinfo("backup.json").file_size > 64 * 1024**2:
            raise ValueError("Манифест резервной копии превышает 64 МБ")
        metadata = json.loads(archive.read("backup.json"))
        if metadata.get("schema_version") != 1:
            raise ValueError("Версия резервной копии не поддерживается")
        project_id = metadata["project_id"]
        if len(project_id) != 32 or any(char not in "0123456789abcdef" for char in project_id):
            raise ValueError("Некорректный идентификатор проекта")
        if session.get(Project, project_id):
            raise ValueError("Проект уже существует; слияние резервных копий не поддерживается")
        allowed_prefix = f"projects/{project_id}/"
        records = metadata["records"]
        ids = {table: {row["id"] for row in rows} for table, rows in records.items()}
        if ids.get("projects") != {project_id}:
            raise ValueError("Состав проекта не соответствует манифесту")
        references = {
            "document_id": "documents",
            "page_id": "pages",
            "job_id": "jobs",
            "source_result_id": "results",
            "lighton_result_id": "results",
            "qwen_result_id": "results",
            "parent_id": "annotations",
            "current_annotation_id": "annotations",
        }
        for rows in records.values():
            for row in rows:
                if row.get("project_id", project_id) != project_id:
                    raise ValueError("В резервной копии смешаны проекты")
                for field, table in references.items():
                    if row.get(field) and row[field] not in ids.get(table, set()):
                        raise ValueError("Связь с объектом вне резервной копии")
                for field in (
                    "image_path",
                    "source_path",
                    "raw_path",
                    "manifest_path",
                    "result_path",
                    "path",
                ):
                    value = row.get(field)
                    if value and (
                        ".." in PurePosixPath(value).parts
                        or not value.startswith(allowed_prefix)
                        or value not in metadata["files"]
                        and field != "path"
                    ):
                        raise ValueError("Путь объекта не принадлежит восстанавливаемому проекту")
        required_size = sum(file["size"] for file in metadata["files"].values())
        if shutil.disk_usage(storage.root).free < required_size + 64 * 1024**2:
            raise ValueError("Недостаточно места для восстановления проекта")
        with tempfile.TemporaryDirectory(dir=storage.root) as temporary:
            temporary_root = Path(temporary)
            for name, expected in metadata["files"].items():
                parts = PurePosixPath(name)
                if (
                    parts.is_absolute()
                    or ".." in parts.parts
                    or not name.startswith(allowed_prefix)
                ):
                    raise ValueError("Небезопасный путь в резервной копии")
                if archive.getinfo(name).file_size != expected["size"]:
                    raise ValueError("Размер файла не соответствует манифесту")
                target = temporary_root / name
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(name) as input_file, target.open("wb") as output:
                    shutil.copyfileobj(input_file, output, length=1024 * 1024)
                if (
                    digest(target) != expected["sha256"]
                    or target.stat().st_size != expected["size"]
                ):
                    raise ValueError("Контрольная сумма файла резервной копии не совпала")
            for row in records.get("datasets", []):
                manifest = json.loads((temporary_root / row["manifest_path"]).read_text())
                if manifest.get("project_id") != project_id or manifest.get("id") != row["id"]:
                    raise ValueError("Версия данных не соответствует проекту")
                for item in manifest["items"]:
                    if item["image"] not in metadata["files"] or item["page_id"] not in ids.get(
                        "pages", set()
                    ):
                        raise ValueError("Версия данных ссылается на отсутствующий объект")
                    if metadata["files"][item["image"]]["sha256"] != item["image_sha256"]:
                        raise ValueError("Версия данных содержит другое изображение")
            # Проверяем только известные таблицы и колонки, не исполняем SQL из архива.
            for model in TABLES:
                columns = {column.name: column for column in model.__table__.columns}
                for row in metadata["records"].get(model.__tablename__, []):
                    values = {
                        key: json.loads(value)
                        if str(columns[key].type) == "JSON" and isinstance(value, str)
                        else value
                        for key, value in row.items()
                        if key in columns
                    }
                    session.add(model(**values))
                session.flush()
            destination = storage.path(f"projects/{project_id}")
            if destination.exists():
                raise ValueError("Каталог проекта уже существует")
            (temporary_root / "projects" / project_id).mkdir(parents=True, exist_ok=True)
            shutil.move(temporary_root / "projects" / project_id, destination)
            try:
                session.commit()
            except Exception:
                shutil.rmtree(destination)
                raise
    return project_id


def delete_project(session, storage, project_id):
    with storage.project_lock(project_id, shared=False):
        if session.scalar(
            select(Job).where(
                Job.project_id == project_id,
                Job.status.not_in({"completed", "failed", "cancelled"}),
            )
        ):
            raise ValueError("Сначала остановите все задания проекта")
        documents = session.scalars(select(Document).where(Document.project_id == project_id)).all()
        page_ids = list(
            session.scalars(select(Page.id).where(Page.document_id.in_([d.id for d in documents])))
        )
        jobs = session.scalars(select(Job).where(Job.project_id == project_id)).all()
        for model in reversed(TABLES[1:]):
            if model is Task:
                statement = select(model).where(Task.job_id.in_([job.id for job in jobs]))
            elif hasattr(model, "project_id"):
                statement = select(model).where(model.project_id == project_id)
            elif model is Page:
                statement = select(model).where(Page.id.in_(page_ids))
            else:
                statement = select(model).where(model.page_id.in_(page_ids))
            for row in session.scalars(statement):
                session.delete(row)
            session.flush()
        session.delete(session.get(Project, project_id))
        session.commit()
        shutil.rmtree(storage.path(f"projects/{project_id}"), ignore_errors=True)
