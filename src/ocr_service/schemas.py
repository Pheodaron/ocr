from typing import Literal

from pydantic import BaseModel, Field

JobStatus = Literal[
    "queued", "processing", "cancelling", "cancelled", "completed", "partial", "failed"
]
PageStatus = Literal["pending", "processing", "cancelled", "completed", "failed"]
TERMINAL_STATUSES = {"cancelled", "completed", "partial", "failed"}


class TextFragment(BaseModel):
    text: str
    bbox: list[float] = Field(min_length=4, max_length=4)
    confidence: float | None = None


class Block(BaseModel):
    kind: str
    text: str = ""
    bbox: list[float] | None = Field(None, min_length=4, max_length=4)


class Cell(BaseModel):
    row: int
    column: int
    text: str
    rowspan: int = 1
    colspan: int = 1
    header: bool = False


class Table(BaseModel):
    bbox: list[float] | None = Field(None, min_length=4, max_length=4)
    html: str
    cells: list[Cell] = Field(default_factory=list)


class PageResult(BaseModel):
    number: int
    status: PageStatus = "completed"
    width: int = 0
    height: int = 0
    dpi: int = 0
    rotation: int = 0
    text: str = ""
    markdown: str = ""
    blocks: list[Block] = Field(default_factory=list)
    fragments: list[TextFragment] = Field(default_factory=list)
    tables: list[Table] = Field(default_factory=list)
    confidence: float | None = None
    needs_review: bool = False
    warnings: list[str] = Field(default_factory=list)
    error: str | None = None
    seconds: float = 0
    attempts: int = 0
    passes: list[dict] = Field(default_factory=list)
    metrics: dict = Field(default_factory=dict)
    image_url: str | None = None


class JobSummary(BaseModel):
    id: str
    filename: str
    sha256: str
    status: JobStatus
    page_count: int
    processed_pages: int
    successful_pages: int
    failed_pages: int
    cancelled_pages: int = 0
    review_pages: int
    created_at: str
    updated_at: str
    error: str | None = None
    backend: str = "paddle"
    comparison_id: str | None = None
    processing_seconds: float = 0
    model_load_seconds: float = 0
    configuration: dict = Field(default_factory=dict)


class Comparison(BaseModel):
    id: str
    jobs: list[JobSummary]


class DocumentResult(BaseModel):
    schema_version: str = "1.0"
    document: JobSummary
    engine: dict
    coordinates: str = (
        "bbox = [x_min, y_min, x_max, y_max], pixels of the orientation-corrected page image"
    )
    pages: list[PageResult]


class WorkerStatus(BaseModel):
    state: str = "offline"
    heartbeat: str | None = None
    job_id: str | None = None
    error: str | None = None
    ready: bool = False
