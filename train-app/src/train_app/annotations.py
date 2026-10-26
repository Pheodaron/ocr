from sqlalchemy.orm import Session

from .grounding import join_blocks, serialize_grounding
from .models import Annotation, Page
from .tables import validate_tables


def save_annotation(session: Session, page: Page, payload) -> Annotation:
    """Каждое сохранение создаёт редакцию; утверждение относится только к ней."""
    if page.current_annotation_id != payload.expected_revision:
        raise ValueError("Редакция уже изменилась. Перезагрузите страницу перед сохранением")
    old = (
        session.get(Annotation, page.current_annotation_id) if page.current_annotation_id else None
    )
    blocks = [item.model_dump() for item in payload.blocks]
    if len({item["id"] for item in blocks}) != len(blocks):
        raise ValueError("Идентификаторы блоков должны быть уникальны")
    content = join_blocks(blocks)
    if content != payload.markdown.strip():
        raise ValueError("Текст страницы не соответствует содержимому блоков")
    if payload.text_approved:
        validate_tables(content)
    if payload.geometry_approved:
        if not payload.text_approved or payload.rejected:
            raise ValueError("Перед геометрией необходимо утвердить текст")
        serialize_grounding(blocks)
    if payload.rejected and (payload.text_approved or payload.geometry_approved):
        raise ValueError("Непригодная страница не может быть утверждена")
    annotation = Annotation(
        page_id=page.id,
        parent_id=old.id if old else None,
        source_result_id=payload.source_result_id,
        image_sha256=page.sha256,
        markdown=content,
        blocks=blocks,
        text_approved=payload.text_approved,
        geometry_approved=payload.geometry_approved,
        rejected=payload.rejected,
        note=payload.note,
    )
    session.add(annotation)
    session.flush()
    page.current_annotation_id = annotation.id
    return annotation
