from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from sqlalchemy import JSON, Boolean, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from .db import Base


def new_id() -> str:
    return uuid4().hex


def timestamp() -> str:
    return datetime.now(UTC).isoformat()


class Record(Base):
    __abstract__ = True
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    created_at: Mapped[str] = mapped_column(default=timestamp)


class Project(Record):
    __tablename__ = "projects"
    title: Mapped[str]
    description: Mapped[str] = mapped_column(Text, default="")
    archived: Mapped[bool] = mapped_column(Boolean, default=False)


class Document(Record):
    __tablename__ = "documents"
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), index=True)
    filename: Mapped[str]
    source_path: Mapped[str]
    sha256: Mapped[str] = mapped_column(index=True)
    page_count: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(default="queued")
    error: Mapped[str | None] = mapped_column(Text)
    split: Mapped[str] = mapped_column(default="unassigned")
    group: Mapped[str] = mapped_column(default="")


class Page(Record):
    __tablename__ = "pages"
    document_id: Mapped[str] = mapped_column(ForeignKey("documents.id"), index=True)
    number: Mapped[int] = mapped_column(Integer)
    image_path: Mapped[str]
    sha256: Mapped[str]
    perceptual_hash: Mapped[str] = mapped_column(default="")
    width: Mapped[int] = mapped_column(Integer)
    height: Mapped[int] = mapped_column(Integer)
    current_annotation_id: Mapped[str | None] = mapped_column(String(32))


class Result(Record):
    __tablename__ = "results"
    page_id: Mapped[str] = mapped_column(ForeignKey("pages.id"), index=True)
    engine: Mapped[str]
    operation_id: Mapped[str] = mapped_column(unique=True)
    status: Mapped[str]
    markdown: Mapped[str] = mapped_column(Text, default="")
    blocks: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    raw_path: Mapped[str | None]
    error: Mapped[str | None] = mapped_column(Text)
    seconds: Mapped[float] = mapped_column(Float, default=0)


class Comparison(Record):
    __tablename__ = "comparisons"
    page_id: Mapped[str] = mapped_column(ForeignKey("pages.id"), index=True)
    lighton_result_id: Mapped[str] = mapped_column(ForeignKey("results.id"))
    qwen_result_id: Mapped[str] = mapped_column(ForeignKey("results.id"))
    agrees: Mapped[bool] = mapped_column(Boolean)
    details: Mapped[dict[str, Any]] = mapped_column(JSON)


class Annotation(Record):
    __tablename__ = "annotations"
    page_id: Mapped[str] = mapped_column(ForeignKey("pages.id"), index=True)
    parent_id: Mapped[str | None] = mapped_column(String(32))
    source_result_id: Mapped[str | None] = mapped_column(ForeignKey("results.id"))
    image_sha256: Mapped[str]
    markdown: Mapped[str] = mapped_column(Text)
    blocks: Mapped[list[dict[str, Any]]] = mapped_column(JSON)
    text_approved: Mapped[bool] = mapped_column(Boolean, default=False)
    geometry_approved: Mapped[bool] = mapped_column(Boolean, default=False)
    rejected: Mapped[bool] = mapped_column(Boolean, default=False)
    note: Mapped[str] = mapped_column(Text, default="")


class Dataset(Record):
    __tablename__ = "datasets"
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), index=True)
    name: Mapped[str]
    manifest_path: Mapped[str]
    sha256: Mapped[str]
    counts: Mapped[dict[str, Any]] = mapped_column(JSON)


class Job(Record):
    __tablename__ = "jobs"
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), index=True)
    kind: Mapped[str]
    status: Mapped[str] = mapped_column(default="queued", index=True)
    updated_at: Mapped[str] = mapped_column(default=timestamp)
    config: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    total: Mapped[int] = mapped_column(Integer, default=0)
    completed: Mapped[int] = mapped_column(Integer, default=0)
    failed: Mapped[int] = mapped_column(Integer, default=0)
    cancel_requested: Mapped[bool] = mapped_column(Boolean, default=False)
    error: Mapped[str | None] = mapped_column(Text)
    result_path: Mapped[str | None]
    hidden: Mapped[bool] = mapped_column(Boolean, default=False)


class Task(Record):
    __tablename__ = "tasks"
    job_id: Mapped[str] = mapped_column(ForeignKey("jobs.id"), index=True)
    page_id: Mapped[str | None] = mapped_column(ForeignKey("pages.id"))
    number: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(default="queued")
    operation_id: Mapped[str] = mapped_column(default=new_id)
    attempt: Mapped[int] = mapped_column(Integer, default=0)
    error: Mapped[str | None] = mapped_column(Text)


class Checkpoint(Record):
    __tablename__ = "checkpoints"
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), index=True)
    job_id: Mapped[str] = mapped_column(ForeignKey("jobs.id"))
    path: Mapped[str]
    step: Mapped[int] = mapped_column(Integer)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
