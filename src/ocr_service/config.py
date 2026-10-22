from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="OCR_", env_file=".env", extra="ignore")

    data_dir: Path = Path("data")
    models_dir: Path = Path("models")
    cpu_threads: int = Field(4, ge=1, le=128)
    dpi: int = Field(250, ge=72, le=600)
    retry_dpi: int = Field(350, ge=72, le=600)
    review_threshold: float = Field(0.75, ge=0, le=1)
    layout_threshold: float = Field(0.5, ge=0, le=1)
    max_upload_mb: int = Field(64, ge=1, le=1024)
    max_pages: int = Field(300, ge=1, le=10000)
    max_page_pixels: int = Field(30_000_000, ge=1_000_000)
    max_page_attempts: int = Field(2, ge=1, le=5)
    poll_seconds: float = Field(1, ge=0.1, le=30)
    worker_stale_seconds: int = Field(45, ge=10)

    @model_validator(mode="after")
    def validate_dpi(self):
        if self.retry_dpi < self.dpi:
            raise ValueError("retry_dpi must be >= dpi")
        return self

    @property
    def db_path(self) -> Path:
        return self.data_dir / "queue.sqlite3"

    def setup(self) -> None:
        (self.data_dir / "documents").mkdir(parents=True, exist_ok=True)

    def recognition_config(self) -> dict:
        return self.model_dump(
            mode="json",
            include={"dpi", "retry_dpi", "review_threshold", "layout_threshold", "max_page_pixels"},
        )
