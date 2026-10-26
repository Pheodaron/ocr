"""Изолированный сервер для проверки интерфейса без GPU и настоящих весов."""

import tempfile
from pathlib import Path

import httpx
from fake_compute import FakeCompute

from train_app.config import Settings
from train_app.main import create_app

original_client = httpx.AsyncClient
fake = FakeCompute()


def fixture_client(*args, **kwargs):
    kwargs["transport"] = httpx.MockTransport(fake)
    return original_client(*args, **kwargs)


httpx.AsyncClient = fixture_client
app = create_app(
    Settings(
        data_dir=Path(tempfile.mkdtemp(prefix="train-app-browser-")),
        frontend_dir=Path("frontend/dist"),
        allowed_origins="http://127.0.0.1:8011",
    )
)
