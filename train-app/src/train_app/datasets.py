import json
import random
import shutil
from pathlib import Path

from sqlalchemy import select

from .grounding import serialize_grounding
from .models import Annotation, Dataset, Document, Job, Page, new_id
from .storage import Storage, atomic_json, digest, read_json
from .tables import validate_tables


def assign_splits(session, project_id: str, fractions: list[float], seed: int) -> dict:
    if len(fractions) != 3 or abs(sum(fractions) - 1) > 0.0001 or min(fractions) < 0:
        raise ValueError("Нужны три неотрицательные доли с суммой 1")
    documents = session.scalars(select(Document).where(Document.project_id == project_id)).all()
    # Общий документ, подтверждённое семейство и точные копии — одна группа.
    parents = {document.id: document.id for document in documents}

    def root(value):
        while parents[value] != value:
            value = parents[value]
        return value

    for index, document in enumerate(documents):
        for other in documents[index + 1 :]:
            if document.sha256 == other.sha256 or (
                document.group and document.group == other.group
            ):
                parents[root(other.id)] = root(document.id)
    groups = {}
    for document in documents:
        groups.setdefault(root(document.id), []).append(document)
    ordered = list(groups.values())
    random.Random(seed).shuffle(ordered)
    counts = {"train": 0, "validation": 0, "test": 0}
    for index, group in enumerate(ordered):
        position = (index + 0.5) / max(len(ordered), 1)
        split = (
            "train"
            if position < fractions[0]
            else "validation"
            if position < sum(fractions[:2])
            else "test"
        )
        for document in group:
            document.split = split
            counts[split] += 1
    return counts


def create_dataset(session, storage: Storage, project_id: str, name: str) -> Dataset:
    rows = session.execute(
        select(Page, Document).join(Document).where(Document.project_id == project_id)
    ).all()
    items = []
    counts = {split: {"pages": 0, "grounding": 0} for split in ("train", "validation", "test")}
    for page, document in rows:
        annotation = (
            session.get(Annotation, page.current_annotation_id)
            if page.current_annotation_id
            else None
        )
        if (
            not annotation
            or not annotation.text_approved
            or annotation.rejected
            or document.split not in counts
        ):
            continue
        if annotation.image_sha256 != page.sha256:
            raise ValueError("Изображение изменилось после утверждения")
        validate_tables(annotation.markdown)
        grounding = serialize_grounding(annotation.blocks) if annotation.geometry_approved else None
        items.append(
            {
                "page_id": page.id,
                "document_id": document.id,
                "filename": document.filename,
                "document_sha256": document.sha256,
                "image": page.image_path,
                "image_sha256": page.sha256,
                "perceptual_hash": page.perceptual_hash,
                "annotation_id": annotation.id,
                "split": document.split,
                "group": document.group,
                "markdown": annotation.markdown,
                "grounding": grounding,
                "blocks": annotation.blocks,
                "width": page.width,
                "height": page.height,
            }
        )
        counts[document.split]["pages"] += 1
        counts[document.split]["grounding"] += bool(grounding)
    if not items:
        raise ValueError("Нет утверждённых и распределённых страниц")
    # Одинаковое изображение не может одновременно быть обучением и тестом.
    for key in ("document_sha256", "image_sha256"):
        splits = {}
        for item in items:
            if item[key] in splits and splits[item[key]] != item["split"]:
                raise ValueError("Одинаковые данные оказались в разных выборках")
            splits[item[key]] = item["split"]
    groups = {}
    for item in items:
        if item["group"]:
            groups.setdefault(item["group"], set()).add(item["split"])
    if any(len(value) > 1 for value in groups.values()):
        raise ValueError("Одно семейство документов находится в разных выборках")
    identifier = new_id()
    path = storage.project_dir(project_id) / "datasets" / identifier / "manifest.json"
    atomic_json(
        path,
        {
            "schema_version": 1,
            "project_id": project_id,
            "id": identifier,
            "name": name,
            "counts": counts,
            "items": items,
        },
    )
    dataset = Dataset(
        id=identifier,
        project_id=project_id,
        name=name,
        manifest_path=storage.relative(path),
        sha256=digest(path),
        counts=counts,
    )
    session.add(dataset)
    return dataset


def training_history(session, storage: Storage, project_id: str) -> dict:
    documents, images, groups = set(), set(), set()
    jobs = session.scalars(
        select(Job).where(Job.project_id == project_id, Job.kind == "train")
    ).all()
    for job in jobs:
        # Начатое обучение считается воздействием, даже если позднее завершилось ошибкой.
        if not job.config.get("training_started"):
            continue
        dataset = session.get(Dataset, job.config.get("dataset_id"))
        if not dataset:
            continue
        for item in read_json(storage.path(dataset.manifest_path))["items"]:
            if item["split"] in {"train", "validation"}:
                documents.add(item["document_sha256"])
                images.add(item["image_sha256"])
                if item["group"]:
                    groups.add(item["group"])
    return {"documents": sorted(documents), "images": sorted(images), "groups": sorted(groups)}


def with_validation(session, storage: Storage, dataset: Dataset) -> Dataset:
    """Производная версия, если пользователь назначил только обучение и тест."""
    if dataset.counts["validation"]["pages"]:
        return dataset
    manifest = read_json(storage.path(dataset.manifest_path))
    train = [item for item in manifest["items"] if item["split"] == "train"]
    parents = {item["document_id"]: item["document_id"] for item in train}

    def find(identifier):
        while parents[identifier] != identifier:
            identifier = parents[identifier]
        return identifier

    owners = {}
    for item in train:
        tokens = [("source", item["document_sha256"]), ("image", item["image_sha256"])]
        if item["group"]:
            tokens.append(("group", item["group"]))
        for token in tokens:
            if token in owners:
                parents[find(item["document_id"])] = find(owners[token])
            owners[token] = item["document_id"]
    groups = sorted({find(item["document_id"]) for item in train})
    if len(groups) < 2:
        raise ValueError(
            "Для независимой валидации нужны хотя бы два разных документа или семейства в обучении"
        )
    random.Random(42).shuffle(groups)
    validation_groups = set(groups[: max(1, round(len(groups) * 0.1))])
    counts = {split: {"pages": 0, "grounding": 0} for split in ("train", "validation", "test")}
    for item in manifest["items"]:
        if item["split"] == "train" and find(item["document_id"]) in validation_groups:
            item["split"] = "validation"
        counts[item["split"]]["pages"] += 1
        counts[item["split"]]["grounding"] += bool(item.get("grounding"))
    identifier = new_id()
    manifest.update(
        id=identifier,
        counts=counts,
        parent_dataset_id=dataset.id,
        name=dataset.name + " / выделена валидация",
        split_derivation={"algorithm": "document-groups-10-percent-1", "seed": 42},
    )
    path = storage.project_dir(dataset.project_id) / "datasets" / identifier / "manifest.json"
    atomic_json(path, manifest)
    derived = Dataset(
        id=identifier,
        project_id=dataset.project_id,
        name=manifest["name"],
        manifest_path=storage.relative(path),
        sha256=digest(path),
        counts=counts,
    )
    session.add(derived)
    return derived


def export_dataset(storage: Storage, dataset: Dataset) -> Path:
    with storage.project_lock(dataset.project_id, shared=True):
        import zipfile

        manifest = read_json(storage.path(dataset.manifest_path))
        target = storage.path(f"exports/dataset-{dataset.id}-{new_id()}.zip")
        target.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(target.with_suffix(".tmp"), "w", zipfile.ZIP_DEFLATED) as archive:
            exported, added = [], set()
            for item in manifest["items"]:
                image_name = f"images/{item['image_sha256']}.png"
                if image_name not in added:
                    archive.write(storage.path(item["image"]), image_name)
                    added.add(image_name)
                exported.append({**item, "image": image_name})
            archive.writestr(
                "manifest.json",
                json.dumps({**manifest, "items": exported}, ensure_ascii=False, indent=2),
            )
            for split in ("train", "validation", "test"):
                examples = []
                for item in exported:
                    if item["split"] != split:
                        continue
                    examples.append(
                        {
                            "page_id": item["page_id"],
                            "image": item["image"],
                            "prompt": "",
                            "answer": item["markdown"],
                        }
                    )
                    if item["grounding"]:
                        examples.append(
                            {
                                "page_id": item["page_id"],
                                "image": item["image"],
                                "prompt": "grounding",
                                "answer": item["grounding"],
                            }
                        )
                archive.writestr(
                    f"{split}.jsonl",
                    "\n".join(json.dumps(row, ensure_ascii=False) for row in examples),
                )
        shutil.move(target.with_suffix(".tmp"), target)
        return target
