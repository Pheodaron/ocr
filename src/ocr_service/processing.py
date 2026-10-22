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
from .schemas import PageResult

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


def process_page(
    engine, source: Path, number: int, directory: Path, settings: Settings
) -> PageResult:
    started = time.monotonic()
    passes = []
    selected = None
    selected_image = None
    dpis = [settings.dpi]
    if settings.retry_dpi > settings.dpi:
        dpis.append(settings.retry_dpi)
    for dpi in dpis:
        try:
            image = render_page(source, number, dpi, settings.max_page_pixels)
            if selected is None:
                save_image(image, directory / f"page-{number}.png")
            prediction = engine.recognize(image, layout_threshold=settings.layout_threshold)
            processed = prediction.image
            raw = json.loads(json.dumps(prediction.raw, default=json_default))
            atomic_json(directory / f"page-{number}-{dpi}.raw.json", raw)
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
            passes.append(
                {
                    "dpi": dpi,
                    "confidence": candidate.confidence,
                    "error": None,
                    "metrics": candidate.metrics,
                }
            )
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
            passes.append({"dpi": dpi, "confidence": None, "error": str(error)[:1000]})
            if selected is not None:
                selected.needs_review = True
                selected.warnings.append(f"Повторный проход {dpi} dpi завершился ошибкой")
    if selected is None:
        raise RuntimeError("; ".join(item["error"] or "" for item in passes))
    selected.passes = passes
    selected.seconds = round(time.monotonic() - started, 3)
    save_image(selected_image, directory / f"page-{number}.png")
    return selected


def process_job(
    db: Database, job: dict, engine, settings: Settings, stopping=lambda: False
) -> None:
    processing_started = time.monotonic()
    job_id = job["id"]
    directory = document_path(settings, job_id)
    source = directory / "source.pdf"
    # Submitted render/quality settings remain stable across worker restarts.
    settings = settings.model_copy(update=json.loads(job["config_json"]))
    metadata = dict(engine.metadata)
    metadata["settings"] = settings.recognition_config()
    db.set_engine(job_id, metadata)
    for page in db.pages(job_id):
        if page.status in {"completed", "failed"}:
            continue
        if stopping():
            db.record_timings(job_id, processing=time.monotonic() - processing_started)
            return
        attempts = db.start_page(job_id, page.number)
        started = time.monotonic()
        try:
            result = process_page(engine, source, page.number, directory, settings)
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
                seconds=round(time.monotonic() - started, 3),
            )
            if (directory / f"page-{page.number}.png").is_file():
                result.image_url = f"/api/documents/{job_id}/pages/{page.number}/image"
        db.save_page(job_id, result)
    db.record_timings(job_id, processing=time.monotonic() - processing_started)
    db.finish_job(job_id)
