"""Local-only PaddleX factories, including the pinned 3.3.13 chart-init workaround.

Each Paddle document runs in its own process. Scope the factory replacement to
construction/prediction and restore it even on exceptions; the API never imports PaddleX.
"""

import logging
from contextlib import contextmanager
from pathlib import Path

from .models import MODEL_ROLES, ModelsUnavailable

LOGGER = logging.getLogger(__name__)


@contextmanager
def local_model_factories(models_dir: Path):
    from paddlex.inference.pipelines.base import BasePipeline

    original = BasePipeline.create_model
    allowed = {name: (models_dir / name).resolve() for name in MODEL_ROLES.values()}

    def create_local_model(pipeline, config, **kwargs):
        name = config.get("model_name")
        # PaddleX 3.3.13 unconditionally creates this model in inintial_predictor,
        # even when use_chart_recognition=False. None is already supported by close().
        if (
            config.get("module_name") == "chart_recognition"
            and getattr(pipeline, "use_chart_recognition", None) is False
        ):
            LOGGER.info("Skipping disabled chart model: %s", name)
            return None
        directory = config.get("model_dir")
        if (
            name not in allowed
            or not directory
            or Path(directory).resolve() != allowed[name]
            or not allowed[name].is_dir()
        ):
            raise ModelsUnavailable(
                f"PaddleX запросил неподготовленную модель {name!r}, model_dir={directory!r}. "
                "Автозагрузка в OCR-процессе запрещена; проверьте локальную конфигурацию."
            )
        return original(pipeline, config, **kwargs)

    BasePipeline.create_model = create_local_model
    try:
        yield
    finally:
        BasePipeline.create_model = original
