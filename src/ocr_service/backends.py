"""Stable, user-visible OCR configuration identities."""

from .catalog import VLM_BACKENDS, backend_label
from .config import Settings

# Keep old result descriptions readable without enabling their inference engines.
DISPLAY_VLM_BACKENDS = VLM_BACKENDS | {"chandra", "lighton", "surya", "jina"}


def configuration_details(backend: str, config: dict) -> dict:
    return {
        "id": backend,
        "label": backend_label(backend),
        "quantization": config.get("dots_quantization", "Q5_K_M")
        if backend == "dots_mocr"
        else config.get("lighton_small_quantization", "Q8_0")
        if backend == "lighton_small"
        else "Q8_0"
        if backend in {"glm_ocr", "lighton", "jina"}
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
        if backend in DISPLAY_VLM_BACKENDS
        else None,
        "task": config.get("glm_task", "text")
        if backend == "glm_ocr"
        else config.get("lighton_small_task", "grounding")
        if backend == "lighton_small"
        else None,
        "render_dpi": config.get("dpi", 250)
        if backend == "paddle"
        else config.get("vlm_dpi", config.get("dpi", 250)),
        "retry_dpi": config.get("retry_dpi", 350) if backend == "paddle" else None,
        "max_input_pixels": min(
            config.get("vlm_max_pixels", 1_500_000),
            config.get("dots_max_pixels", config.get("vlm_max_pixels", 1_500_000)),
        )
        if backend == "dots_mocr"
        else config.get("vlm_max_pixels", 1_500_000)
        if backend in DISPLAY_VLM_BACKENDS
        else None,
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
    if backend in VLM_BACKENDS:
        from .vlm import VLMEngine

        return VLMEngine(settings, backend)
    raise ValueError(f"Unknown OCR backend: {backend}")
