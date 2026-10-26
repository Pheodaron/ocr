import logging
import math
import os
from pathlib import Path

import pymupdf
from PIL import Image, ImageOps
from sqlalchemy import select

from .models import Document, Page, new_id
from .storage import Storage, digest

logger = logging.getLogger(__name__)
Image.MAX_IMAGE_PIXELS = 40_000_000


def inspect_document(path: Path, suffix: str, max_pages: int) -> int:
    if suffix == ".pdf":
        with pymupdf.open(path) as document:
            if document.needs_pass:
                raise ValueError("PDF защищён паролем")
            count = len(document)
    elif suffix in {".png", ".jpg", ".jpeg"}:
        with Image.open(path) as image:
            image.verify()
        count = 1
    else:
        raise ValueError("Поддерживаются PDF, PNG и JPEG")
    if not 1 <= count <= max_pages:
        raise ValueError(f"Документ должен содержать от 1 до {max_pages} страниц")
    return count


def image_hash(image: Image.Image) -> str:
    """64-битный dHash: только кандидат похожести, не основание удаления."""
    pixels = list(image.convert("L").resize((9, 8)).getdata())
    value = 0
    for row in range(8):
        for column in range(8):
            value = (value << 1) | (pixels[row * 9 + column] > pixels[row * 9 + column + 1])
    return f"{value:016x}"


def render_page(document: Document, number: int, storage: Storage, settings) -> Page:
    source = storage.path(document.source_path)
    if source.suffix.lower() == ".pdf":
        with pymupdf.open(source) as pdf:
            page = pdf[number - 1]
            width, height = page.rect.width, page.rect.height
            scale = settings.render_dpi / 72
            if width * height * scale * scale > settings.max_page_pixels:
                raise ValueError(
                    "Страница превышает лимит пикселей; измените явный профиль подготовки"
                )
            pixmap = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), alpha=False)
            image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
    else:
        with Image.open(source) as original:
            image = ImageOps.exif_transpose(original).convert("RGB")
    if image.width * image.height > settings.max_page_pixels:
        raise ValueError("Изображение превышает лимит пикселей")
    page_id = new_id()
    path = storage.project_dir(document.project_id) / "images" / f"{page_id}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    image.save(temporary, format="PNG")
    os.replace(temporary, path)
    return Page(
        id=page_id,
        document_id=document.id,
        number=number,
        image_path=storage.relative(path),
        sha256=digest(path),
        perceptual_hash=image_hash(image),
        width=image.width,
        height=image.height,
    )


def duplicate_candidates(session, project_id: str) -> list[dict]:
    pages = session.execute(
        select(Page, Document).join(Document).where(Document.project_id == project_id)
    ).all()
    pairs = []
    for index, (page, document) in enumerate(pages):
        for other, other_document in pages[index + 1 :]:
            if document.id == other_document.id:
                continue
            exact = page.sha256 == other.sha256
            distance = (
                int(page.perceptual_hash or "0", 16) ^ int(other.perceptual_hash or "0", 16)
            ).bit_count()
            if exact or distance <= 3:
                pairs.append(
                    {
                        "left": page.id,
                        "right": other.id,
                        "left_document": document.id,
                        "right_document": other_document.id,
                        "exact": exact,
                        "hash_distance": distance,
                        "cross_split": document.split != other_document.split,
                    }
                )
    return pairs


def model_image(path: Path, longest_edge: int) -> Image.Image:
    with Image.open(path) as original:
        image = original.convert("RGB")
    ratio = min(1.0, longest_edge / max(image.size))
    if ratio < 1:
        image = image.resize(
            (math.ceil(image.width * ratio), math.ceil(image.height * ratio)),
            Image.Resampling.LANCZOS,
        )
    return image
