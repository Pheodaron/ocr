from train_app.datasets import with_validation
from train_app.models import Dataset, new_id
from train_app.storage import atomic_json, digest, read_json


def test_validation_preserves_source_and_keeps_duplicate_images_together(app, project):
    storage = app.state.storage
    identifier = new_id()
    items = [
        {
            "document_id": document,
            "document_sha256": document,
            "image_sha256": image,
            "split": "train",
            "group": "",
            "grounding": None,
        }
        for document, image in [("a", "same"), ("b", "same"), ("c", "other")]
    ]
    counts = {
        split: {"pages": count, "grounding": 0}
        for split, count in [("train", 3), ("validation", 0), ("test", 0)]
    }
    path = storage.project_dir(project["id"]) / "datasets" / identifier / "manifest.json"
    atomic_json(path, {"id": identifier, "name": "Источник", "items": items, "counts": counts})
    original_hash = digest(path)
    with app.state.database() as session:
        source = Dataset(
            id=identifier,
            project_id=project["id"],
            name="Источник",
            manifest_path=storage.relative(path),
            sha256=original_hash,
            counts=counts,
        )
        session.add(source)
        session.flush()
        derived = with_validation(session, storage, source)
        manifest = read_json(storage.path(derived.manifest_path))
        assert derived.id != source.id
        assert manifest["parent_dataset_id"] == source.id
        splits = {item["document_id"]: item["split"] for item in manifest["items"]}
        assert splits["a"] == splits["b"] != splits["c"]
        assert set(splits.values()) == {"train", "validation"}
    assert digest(path) == original_hash
