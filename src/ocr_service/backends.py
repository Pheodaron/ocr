"""Stable, user-visible OCR configuration identities."""

from .catalog import LABELS, VLM_BACKENDS
from .config import Settings


def configuration_details(backend: str, config: dict) -> dict:
    return {
        "id": backend,
        "label": LABELS.get(backend, backend),
        "quantization": config.get("dots_quantization", "Q5_K_M")
        if backend == "dots_mocr"
        else "Q8_0"
        if backend in {"glm_ocr", "lighton", "lighton_small", "jina"}
        else "Q5_K_M"
        if backend == "chandra"
        else "FP32"
        if backend in {"kdl", "falcon", "nemotron"}
        else "F16"
        if backend == "surya"
        else None,
        "context": config.get("dots_context", 32768)
        if backend == "dots_mocr"
        else config.get("glm_context", 16384)
        if backend == "glm_ocr"
        else 8192
        if backend == "kdl"
        else config.get("vlm_context", 16384)
        if backend in VLM_BACKENDS
        else None,
        "task": config.get("glm_task", "text") if backend == "glm_ocr" else None,
    }


def configurations(settings: Settings) -> list[dict]:
    return [
        configuration_details(backend, settings.recognition_config())
        for backend in settings.backends
    ]


def create_engine(backend: str, settings: Settings):
    if backend == "paddle":
        from .engine import PaddleEngine

        return PaddleEngine(settings)
    if backend == "kdl":
        from .kdl import KDLEngine

        return KDLEngine(settings)
    if backend == "falcon":
        from .falcon import FalconEngine

        return FalconEngine(settings)
    if backend == "nemotron":
        from .nemotron import NemotronEngine

        return NemotronEngine(settings)
    if backend in VLM_BACKENDS:
        from .vlm import VLMEngine

        return VLMEngine(settings, backend)
    raise ValueError(f"Unknown OCR backend: {backend}")
