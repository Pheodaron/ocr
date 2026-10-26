from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="TRAIN_", env_file=".env", extra="ignore")

    data_dir: Path = Path("data")
    models_dir: Path = Path("models")
    frontend_dir: Path = Path("frontend/dist")
    lighton_url: str = "http://lighton4b:8090"
    trainer_url: str = "http://trainer:8090"
    qwen_url: str = "http://qwen:8080"
    qwen_model: str = "qwen-ocr"
    max_upload_mb: int = 500
    max_document_pages: int = 2000
    max_page_pixels: int = 30_000_000
    render_dpi: int = 250
    model_longest_edge: int = 2048
    max_output_tokens: int = 4096
    inference_timeout: int = 3600
    allowed_origins: str = (
        "http://localhost:8001,http://127.0.0.1:8001,http://localhost:5173,http://127.0.0.1:5173"
    )
    dispatcher: bool = True

    @property
    def database_url(self) -> str:
        return f"sqlite:///{self.data_dir.resolve() / 'workbench.sqlite'}"
