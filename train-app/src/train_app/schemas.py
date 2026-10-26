from typing import Annotated, Literal

from pydantic import BaseModel, Field

Identifier = Annotated[str, Field(pattern=r"^[a-f0-9]{32}$")]


class ProjectInput(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    description: str = ""
    archived: bool = False


class SplitInput(BaseModel):
    document_ids: list[Identifier]
    split: Literal["train", "validation", "test", "unassigned"] | None = None
    group: str | None = None


class AutoSplitInput(BaseModel):
    fractions: list[float] = [0.8, 0.1, 0.1]
    seed: int = 42


class Block(BaseModel):
    id: str = Field(min_length=1, max_length=64)
    type: str = Field(default="text", pattern=r"^[\w+\-]+$")
    content: str
    bbox: list[float] | None = None


class AnnotationInput(BaseModel):
    expected_revision: Identifier | None = None
    source_result_id: Identifier | None = None
    markdown: str
    blocks: list[Block] = Field(min_length=1)
    text_approved: bool = False
    geometry_approved: bool = False
    rejected: bool = False
    note: str = ""


class JobInput(BaseModel):
    kind: Literal["lighton", "qwen", "compare", "train", "evaluate", "probe", "export"]
    document_ids: list[Identifier] = []
    page_ids: list[Identifier] = []
    mode: Literal["missing", "failed", "all"] = "missing"
    config: dict = {}


class DatasetInput(BaseModel):
    name: str = Field(min_length=1, max_length=200)


class TrainingConfig(BaseModel):
    dataset_id: Identifier
    mode: Literal["ocr", "mixed"] = "ocr"
    epochs: int = Field(default=3, ge=1, le=100)
    learning_rate: float = Field(default=0.0001, gt=0, le=0.01)
    gradient_accumulation: int = Field(default=8, ge=1, le=256)
    rank: int = Field(default=16, ge=1, le=128)
    save_steps: int = Field(default=50, ge=1)
    validation_steps: int = Field(default=50, ge=1)
    grounding_fraction: float = Field(default=0.2, gt=0, lt=1)
    longest_edge: int = Field(default=2048, ge=512, le=4096)
    max_tokens: int = Field(default=16384, ge=512, le=65536)
    max_output_tokens: int = Field(default=8192, ge=128, le=32768)
    seed: int = 42
    train_vision: bool = False
    resume_checkpoint_id: Identifier | None = None


class InferenceConfig(BaseModel):
    dataset_id: Identifier | None = None
    checkpoint_id: Identifier | None = None
    prompt: Literal["", "grounding"] = ""
    longest_edge: int = Field(default=2048, ge=512, le=4096)
    max_output_tokens: int = Field(default=8192, ge=128, le=32768)
