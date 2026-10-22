"""Exercise real PaddleOCR/PaddleX orchestration, replacing only weight predictors."""

from pathlib import Path
from unittest.mock import Mock

import pytest

from ocr_service.engine import PaddleEngine
from ocr_service.models import MODEL_ROLES


def test_real_paddle_constructor_never_loads_disabled_chart_or_remote_models(settings, monkeypatch):
    monkeypatch.setenv("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    monkeypatch.setenv("PADDLE_PDX_CACHE_HOME", str(settings.data_dir / "paddle-cache"))
    pytest.importorskip("paddleocr")
    import paddlex.inference

    from ocr_service import engine

    for name in MODEL_ROLES.values():
        (settings.models_dir / name).mkdir(parents=True)
    monkeypatch.setattr(engine, "verify_models", lambda _: {"fixture": True})
    requested = []

    def predictor(*, model_name, model_dir=None, **kwargs):
        assert model_name in MODEL_ROLES.values(), f"Unexpected model download: {model_name}"
        assert Path(model_dir) == (settings.models_dir / model_name).resolve()
        requested.append(model_name)
        return Mock(model_name=model_name)

    monkeypatch.setattr(paddlex.inference, "create_predictor", predictor)
    instance = PaddleEngine(settings)
    pipeline = instance.pipeline.paddlex_pipeline
    assert pipeline.use_chart_recognition is False
    assert pipeline.chart_recognition_model is None
    assert set(requested) == set(MODEL_ROLES.values())
    # The lazy OCR fallback and orientation path must obey the same local-model rule.
    table = pipeline.table_recognition_pipeline
    table.create_pipeline(table.general_ocr_config_bak)
    table.create_model(table.table_orientation_classify_config)
    assert len(requested) == 12


def test_factory_rejects_missing_local_paths_and_restores_after_error(settings, monkeypatch):
    monkeypatch.setenv("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    monkeypatch.setenv("PADDLE_PDX_CACHE_HOME", str(settings.data_dir / "paddle-cache"))
    pytest.importorskip("paddleocr")
    from paddlex.inference.pipelines.base import BasePipeline

    from ocr_service.models import ModelsUnavailable
    from ocr_service.paddle_runtime import local_model_factories

    original = BasePipeline.create_model
    forbidden = Mock(side_effect=AssertionError("Remote predictor resolution must not run"))
    monkeypatch.setattr("paddlex.inference.create_predictor", forbidden)
    pipeline = Mock(use_chart_recognition=False)
    with pytest.raises(ModelsUnavailable, match="Автозагрузка"):
        with local_model_factories(settings.models_dir):
            assert (
                BasePipeline.create_model(
                    pipeline,
                    {
                        "module_name": "chart_recognition",
                        "model_name": "PP-Chart2Table",
                    },
                )
                is None
            )
            BasePipeline.create_model(
                pipeline,
                {
                    "model_name": MODEL_ROLES["recognition"],
                    "model_dir": None,
                },
            )
    assert BasePipeline.create_model is original
    forbidden.assert_not_called()
