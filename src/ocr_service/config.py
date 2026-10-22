from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from .catalog import LABELS


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="OCR_", env_file=".env", extra="ignore")

    data_dir: Path = Path("data")
    models_dir: Path = Path("models")
    cpu_threads: int = Field(4, ge=1, le=128)
    dpi: int = Field(250, ge=72, le=600)
    retry_dpi: int = Field(350, ge=72, le=600)
    vlm_dpi: int = Field(150, ge=72, le=600)
    review_threshold: float = Field(0.75, ge=0, le=1)
    layout_threshold: float = Field(0.5, ge=0, le=1)
    max_upload_mb: int = Field(64, ge=1, le=1024)
    max_pages: int = Field(300, ge=1, le=10000)
    max_page_pixels: int = Field(30_000_000, ge=1_000_000)
    max_page_attempts: int = Field(2, ge=1, le=5)
    poll_seconds: float = Field(1, ge=0.1, le=30)
    worker_stale_seconds: int = Field(45, ge=10)
    enabled_backends: str = ",".join(LABELS)
    worker_backend: str = "paddle"
    dots_quantization: str = "Q5_K_M"
    dots_context: int = Field(32768, ge=2048, le=131072)
    glm_context: int = Field(16384, ge=2048, le=131072)
    glm_task: str = "text"
    vlm_max_pixels: int = Field(1_500_000, ge=65_536, le=30_000_000)
    dots_max_pixels: int = Field(1_000_000, ge=65_536, le=30_000_000)
    vlm_max_tokens: int = Field(8192, ge=128, le=65536)
    vlm_timeout_seconds: int = Field(3600, ge=10)
    vlm_startup_seconds: int = Field(300, ge=10)
    llama_server: str = "llama-server"
    llama_port: int = Field(8081, ge=1024, le=65535)
    kv_cache_type: str = "q8_0"
    vlm_context: int = Field(16384, ge=2048, le=32768)
    kdl_max_regions: int = Field(150, ge=1, le=1000)
    kdl_max_pixels: int = Field(1_003_520, ge=3136, le=4_014_080)

    @model_validator(mode="after")
    def validate_dpi(self):
        if self.retry_dpi < self.dpi:
            raise ValueError("retry_dpi must be >= dpi")
        if not self.backends or set(self.backends) - LABELS.keys():
            raise ValueError(f"enabled_backends: use {','.join(LABELS)}")
        if self.worker_backend not in LABELS:
            raise ValueError("Unknown worker_backend")
        if self.dots_quantization not in {"Q5_K_M", "Q8_0", "BF16"}:
            raise ValueError("dots_quantization: use Q5_K_M, Q8_0 or BF16")
        if self.glm_task not in {"text", "table"}:
            raise ValueError("glm_task: use text or table")
        if self.kv_cache_type not in {"f16", "q8_0"}:
            raise ValueError("kv_cache_type: use f16 or q8_0")
        return self

    @property
    def backends(self) -> list[str]:
        return list(
            dict.fromkeys(item.strip() for item in self.enabled_backends.split(",") if item.strip())
        )

    @property
    def db_path(self) -> Path:
        return self.data_dir / "queue.sqlite3"

    def setup(self) -> None:
        (self.data_dir / "documents").mkdir(parents=True, exist_ok=True)

    def render_dpis(self, backend: str) -> list[int]:
        if backend != "paddle":
            return [self.vlm_dpi]
        return list(dict.fromkeys([self.dpi, self.retry_dpi]))

    def vlm_image_budget(self, backend: str) -> int:
        if backend == "dots_mocr":
            return min(self.vlm_max_pixels, self.dots_max_pixels)
        return self.vlm_max_pixels

    def restore_recognition_config(self, config: dict) -> "Settings":
        config = dict(config)
        # Jobs submitted before backend-specific resolution keep their original input size.
        config.setdefault("vlm_dpi", config.get("dpi", self.dpi))
        config.setdefault("dots_max_pixels", config.get("vlm_max_pixels", self.vlm_max_pixels))
        return Settings.model_validate({**self.model_dump(), **config})

    def recognition_config(self) -> dict:
        return self.model_dump(
            mode="json",
            include={
                "dpi",
                "retry_dpi",
                "vlm_dpi",
                "review_threshold",
                "layout_threshold",
                "max_page_pixels",
                "cpu_threads",
                "dots_quantization",
                "dots_context",
                "glm_context",
                "glm_task",
                "vlm_max_tokens",
                "vlm_max_pixels",
                "dots_max_pixels",
                "kv_cache_type",
                "vlm_timeout_seconds",
                "vlm_context",
                "kdl_max_regions",
                "kdl_max_pixels",
            },
        )
