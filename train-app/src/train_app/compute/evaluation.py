import gc
import os
from pathlib import Path

from train_app.grounding import parse_grounding
from train_app.metrics import aggregate, compare, geometry_metrics
from train_app.storage import atomic_json, read_json
from train_app.tables import validate_tables

from .native import NativeModel


def evaluate(payload, storage, cancelled, progress):
    import torch

    manifest = read_json(storage.path(payload["manifest"]))
    items = [item for item in manifest["items"] if item["split"] == "test"]
    if not items:
        raise ValueError("Нет тестовых страниц")
    history = payload.get("training_history", {})
    excluded, included = [], []
    for item in items:
        contaminated = (
            item["document_sha256"] in history.get("documents", [])
            or item["image_sha256"] in history.get("images", [])
            or bool(item["group"] and item["group"] in history.get("groups", []))
        )
        if contaminated:
            excluded.append(
                {
                    "page_id": item["page_id"],
                    "reason": "Документ или семейство использовались в обучении или валидации",
                }
            )
        else:
            validate_tables(item["markdown"])
            included.append(item)
    if not included:
        raise ValueError("Не осталось независимых тестовых страниц")
    variants = [("baseline", None)]
    if payload.get("checkpoint"):
        variants.append(("trained", storage.path(payload["checkpoint"])))
    reports = {}
    for label, adapter in variants:
        native = NativeModel(
            Path(os.environ.get("TRAIN_STUDENT_PATH", "/models/lighton08")), adapter
        )
        pages = []
        for index, item in enumerate(included):
            if cancelled():
                break
            progress(
                {"stage": "evaluation", "model": label, "page": index + 1, "total": len(included)}
            )
            output, error = {}, None
            try:
                output = native.recognize(
                    storage.path(item["image"]),
                    payload.get("prompt", ""),
                    payload.get("longest_edge", 2048),
                    payload.get("max_output_tokens", 8192),
                    cancelled,
                )
                error = output.get("error")
                markdown = parse_grounding(output.get("text", ""))[0] if not error else ""
            except Exception as exception:
                error, markdown = str(exception), ""
            geometry = None
            if payload.get("prompt") == "grounding" and item.get("grounding"):
                predicted_blocks = parse_grounding(output.get("text", ""))[1] if not error else []
                geometry = geometry_metrics(item["blocks"], predicted_blocks)
            pages.append(
                {
                    "page_id": item["page_id"],
                    "document_id": item["document_id"],
                    "expected": item["markdown"],
                    "markdown": markdown,
                    "error": error,
                    "metrics": compare(item["markdown"], markdown),
                    "geometry": geometry,
                    "output": output,
                }
            )
            atomic_json(
                storage.path(payload["output_dir"]) / f"evaluation-{label}.json",
                {"pages": pages, "complete": False},
            )
        reports[label] = {
            "summary": aggregate(pages),
            "pages": pages,
            "complete": len(pages) == len(included),
            "independent": True,
        }
        del native
        gc.collect()
        torch.cuda.empty_cache()
    return {
        "models": reports,
        "excluded": excluded,
        "dataset_id": manifest["id"],
        "configuration": payload,
        "complete": not cancelled() and all(report["complete"] for report in reports.values()),
    }
