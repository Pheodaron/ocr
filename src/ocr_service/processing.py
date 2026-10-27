import json
import logging
import math
import os
import time
from pathlib import Path

import pymupdf

from .adapter import adapt_prediction
from .config import Settings
from .db import Database
from .engine import InferenceProcessError
from .schemas import PageResult
from .telemetry import ResourceMonitor, timed_stage

LOGGER = logging.getLogger(__name__)


def document_path(settings: Settings, job_id: str) -> Path:
    return settings.data_dir / "documents" / job_id


def validate_pdf(path: Path, max_pages: int) -> int:
    try:
        with pymupdf.open(path) as document:
            if not document.is_pdf:
                raise ValueError("Файл не является PDF")
            if document.needs_pass:
                raise ValueError("PDF защищён паролем; загрузите документ без пароля")
            if not 0 < len(document) <= max_pages:
                raise ValueError(f"PDF должен содержать от 1 до {max_pages} страниц")
            return len(document)
    except (pymupdf.FileDataError, pymupdf.EmptyFileError) as error:
        raise ValueError(
            "Не удалось открыть PDF: файл повреждён или имеет неверный формат"
        ) from error


def render_page(source: Path, number: int, dpi: int, max_pixels: int):
    import numpy as np

    with pymupdf.open(source) as document:
        page = document.load_page(number - 1)
        width = math.ceil(page.rect.width * dpi / 72)
        height = math.ceil(page.rect.height * dpi / 72)
        if width <= 0 or height <= 0 or width * height > max_pixels:
            raise ValueError(f"Страница превышает лимит {max_pixels:,} пикселей при {dpi} dpi")
        pixmap = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csRGB, alpha=False)
        rgb = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(pixmap.height, pixmap.width, 3)
        return np.ascontiguousarray(rgb[:, :, ::-1])


def save_image(image, path: Path) -> None:
    height, width = image.shape[:2]
    rgb = image[:, :, ::-1].tobytes()
    pixmap = pymupdf.Pixmap(pymupdf.csRGB, width, height, rgb, False)
    temporary = path.with_suffix(".tmp.png")
    pixmap.save(temporary)
    os.replace(temporary, path)


def json_default(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Unsupported result type: {type(value).__name__}")


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, default=json_default), encoding="utf-8"
    )
    os.replace(temporary, path)


class PageProcessingError(RuntimeError):
    def __init__(self, passes):
        self.passes = passes
        super().__init__("; ".join(item["error"] or "" for item in passes))


def process_page(
    engine,
    source: Path,
    number: int,
    directory: Path,
    settings: Settings,
    *,
    backend: str | None = None,
) -> PageResult:
    started = time.monotonic()
    passes = []
    selected = None
    selected_image = None
    backend = backend or getattr(engine, "backend", settings.worker_backend)
    dpis = settings.render_dpis(backend)
    for dpi in dpis:
        attempt = {"dpi": dpi, "confidence": None, "error": None, "stages": []}
        passes.append(attempt)
        stages = attempt["stages"]
        try:
            with timed_stage(
                stages,
                "render_pdf",
                renderer="PyMuPDF",
                page=number,
                dpi=dpi,
                colorspace="RGB",
                alpha=False,
                max_pixels=settings.max_page_pixels,
                output="BGR uint8 array",
            ) as details:
                image = render_page(source, number, dpi, settings.max_page_pixels)
                details.update(width=image.shape[1], height=image.shape[0])
            if selected is None:
                with timed_stage(stages, "save_input_image", format="PNG"):
                    save_image(image, directory / f"page-{number}.png")
            with timed_stage(
                stages, "recognize", backend=backend, layout_threshold=settings.layout_threshold
            ):
                prediction = engine.recognize(image, layout_threshold=settings.layout_threshold)
            processed = prediction.image
            with timed_stage(stages, "save_raw_result", format="JSON"):
                raw = json.loads(json.dumps(prediction.raw, default=json_default))
                atomic_json(directory / f"page-{number}-{dpi}.raw.json", raw)
            with timed_stage(stages, "parse_result", format=prediction.format):
                if prediction.format == "kdl":
                    from .vlm_adapter import adapt_kdl

                    candidate = adapt_kdl(raw, number, dpi, processed.shape[1], processed.shape[0])
                elif prediction.format == "vlm":
                    from .vlm_adapter import adapt_vlm

                    candidate = adapt_vlm(raw, number, dpi, processed.shape[1], processed.shape[0])
                else:
                    candidate = adapt_prediction(
                        raw,
                        number,
                        dpi,
                        processed.shape[1],
                        processed.shape[0],
                        settings.review_threshold,
                    )
            candidate.metrics.update(prediction.metrics)
            attempt.update(confidence=candidate.confidence, metrics=dict(candidate.metrics))
            # Keep the retry only when confidence improves without substantial text loss.
            if (
                selected is None
                or (
                    (candidate.confidence or 0) > (selected.confidence or 0)
                    and len(candidate.text) >= len(selected.text) * 0.8
                )
                or (not selected.text.strip() and candidate.text.strip())
            ):
                if selected is not None and candidate.text.split() != selected.text.split():
                    candidate.warnings.append("Текст повторного прохода отличается от первого")
                    candidate.needs_review = True
                selected, selected_image = candidate, processed
            if not candidate.needs_review or not getattr(engine, "retry_on_review", True):
                break
        except Exception as error:
            LOGGER.exception("Page %s failed at %s dpi", number, dpi)
            attempt["error"] = str(error)[:1000]
            if selected is not None:
                selected.needs_review = True
                selected.warnings.append(f"Повторный проход {dpi} dpi завершился ошибкой")
            if isinstance(error, InferenceProcessError):
                # A failed server must be restarted for the next page, not called
                # again with an even larger image in the current page's DPI retry.
                break
    if selected is None:
        raise PageProcessingError(passes)
    selected.passes = passes
    finalization = []
    with timed_stage(finalization, "save_result_image", format="PNG", dpi=selected.dpi):
        save_image(selected_image, directory / f"page-{number}.png")
    selected.metrics["finalization"] = finalization
    selected.seconds = round(time.monotonic() - started, 6)
    return selected


def process_job(
    db: Database, job: dict, engine, settings: Settings, stopping=lambda: False
) -> None:
    processing_started = time.monotonic()
    job_id = job["id"]
    directory = document_path(settings, job_id)
    source = directory / "source.pdf"
    # Submitted render/quality settings remain stable across worker restarts.
    settings = settings.restore_recognition_config(json.loads(job["config_json"]))
    metadata = dict(engine.metadata)
    metadata["settings"] = settings.recognition_config()
    db.set_engine(job_id, metadata)
    for page in db.pages(job_id):
        if page.status in {"completed", "failed"}:
            continue
        if stopping() or db.cancellation_requested(job_id):
            db.record_timings(job_id, processing=time.monotonic() - processing_started)
            return
        attempts = db.start_page(job_id, page.number)
        if attempts is None:
            db.record_timings(job_id, processing=time.monotonic() - processing_started)
            return
        started = time.monotonic()
        with ResourceMonitor() as monitor:
            try:
                result = process_page(
                    engine, source, page.number, directory, settings, backend=job["backend"]
                )
                result.attempts = attempts
                result.image_url = f"/api/documents/{job_id}/pages/{page.number}/image"
            except Exception as error:
                # Page failures are durable results, never an early return from the document.
                LOGGER.exception("Document %s page %s failed", job_id, page.number)
                result = PageResult(
                    number=page.number,
                    status="failed",
                    error=str(error)[:2000],
                    needs_review=True,
                    attempts=attempts,
                    seconds=round(time.monotonic() - started, 6),
                    passes=getattr(error, "passes", []),
                )
                if (directory / f"page-{page.number}.png").is_file():
                    result.image_url = f"/api/documents/{job_id}/pages/{page.number}/image"
        result.metrics["resources"] = monitor.metrics()
        db.save_page(job_id, result)
    db.record_timings(job_id, processing=time.monotonic() - processing_started)
    if not db.cancellation_requested(job_id):
        db.finish_job(job_id)
