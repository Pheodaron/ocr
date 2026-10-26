"""Объединение LoRA, экспорт GGUF и проверка Q4 на CPU."""

import gc
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx

from train_app.documents import model_image
from train_app.metrics import aggregate, compare
from train_app.storage import atomic_json, digest, read_json

from .native import NativeModel


def command(arguments, cancelled):
    process = subprocess.Popen([str(part) for part in arguments])
    while process.poll() is None:
        if cancelled():
            process.terminate()
            process.wait(timeout=20)
            raise RuntimeError("Экспорт отменён")
        time.sleep(0.5)
    if process.returncode:
        raise RuntimeError(f"Команда экспорта завершилась с кодом {process.returncode}")


def cpu_evaluate(model, projector, items, payload, storage, cancelled):
    import base64
    import io

    server = Path(os.environ.get("TRAIN_LLAMA_SERVER", "/opt/llama/bin/llama-server"))
    process = subprocess.Popen(
        [
            str(server),
            "-m",
            str(model),
            "--mmproj",
            str(projector),
            "--device",
            "none",
            "--gpu-layers",
            "0",
            "--no-mmproj-offload",
            "--fit",
            "off",
            "-c",
            "16384",
            "-np",
            "1",
            "-t",
            "8",
            "--host",
            "127.0.0.1",
            "--port",
            "8099",
        ]
    )
    pages = []
    try:
        with httpx.Client(trust_env=False, timeout=3600) as client:
            for _ in range(180):
                if process.poll() is not None:
                    raise RuntimeError("CPU llama-server не запустился")
                try:
                    if client.get("http://127.0.0.1:8099/health", timeout=2).is_success:
                        break
                except httpx.HTTPError:
                    pass
                time.sleep(1)
            else:
                raise RuntimeError("Превышено время запуска CPU llama-server")
            for item in items:
                if cancelled():
                    break
                image = model_image(storage.path(item["image"]), payload.get("longest_edge", 2048))
                buffer = io.BytesIO()
                image.save(buffer, format="PNG")
                uri = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
                try:
                    response = client.post(
                        "http://127.0.0.1:8099/v1/chat/completions",
                        json={
                            "messages": [
                                {
                                    "role": "user",
                                    "content": [{"type": "image_url", "image_url": {"url": uri}}],
                                }
                            ],
                            "temperature": 0,
                            "seed": 42,
                            "max_tokens": 8192,
                            "chat_template_kwargs": {"enable_thinking": False},
                        },
                    )
                    response.raise_for_status()
                    raw = response.json()
                    choice = raw["choices"][0]
                    text = choice["message"].get("content") or ""
                    if choice.get("finish_reason") != "stop":
                        raise ValueError("Экспортированная модель не завершила ответ")
                    pages.append(
                        {
                            "page_id": item["page_id"],
                            "markdown": text,
                            "metrics": compare(item["markdown"], text),
                            "raw": raw,
                        }
                    )
                except Exception as error:
                    pages.append(
                        {
                            "page_id": item["page_id"],
                            "error": str(error),
                            "metrics": compare(item["markdown"], ""),
                        }
                    )
    finally:
        process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
    return {"pages": pages, "summary": aggregate(pages), "complete": len(pages) == len(items)}


def export_model(payload, storage, cancelled, progress):
    import torch

    output = storage.path(payload["output_dir"]) / "export"
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = storage.path(payload["checkpoint"])
    native = NativeModel(
        Path(os.environ.get("TRAIN_STUDENT_PATH", "/models/lighton08")), checkpoint
    )
    progress({"stage": "merge"})
    merged = output / "merged"
    native.model = native.model.merge_and_unload()
    native.model.save_pretrained(merged, safe_serialization=True)
    native.processor.save_pretrained(merged)
    manifest = (
        read_json(storage.path(payload["manifest"])) if payload.get("manifest") else {"items": []}
    )
    history = payload.get("training_history", {})
    test = [
        item
        for item in manifest["items"]
        if item["split"] == "test"
        and item["document_sha256"] not in history.get("documents", [])
        and item["image_sha256"] not in history.get("images", [])
        and not (item["group"] and item["group"] in history.get("groups", []))
    ]
    full_pages = []
    for item in test:
        if cancelled():
            raise RuntimeError("Экспорт отменён")
        answer = native.recognize(storage.path(item["image"]), "", 2048, 8192, cancelled)
        full_pages.append(
            {
                "page_id": item["page_id"],
                "error": answer.get("error"),
                "metrics": compare(
                    item["markdown"], answer["text"] if not answer.get("error") else ""
                ),
            }
        )
    del native
    gc.collect()
    torch.cuda.empty_cache()
    converter = (
        Path(os.environ.get("TRAIN_LLAMA_SOURCE", "/opt/llama/source")) / "convert_hf_to_gguf.py"
    )
    quantizer = Path(os.environ.get("TRAIN_LLAMA_QUANTIZE", "/opt/llama/bin/llama-quantize"))
    full = output / "lighton08-F16.gguf"
    quantized = output / "lighton08-Q4_K_M.gguf"
    progress({"stage": "convert_gguf"})
    command([sys.executable, converter, merged, "--outfile", full, "--outtype", "f16"], cancelled)
    # Визуальная часть экспортируется из той же объединённой модели, включая изменённые веса.
    command(
        [
            sys.executable,
            converter,
            merged,
            "--mmproj",
            "--outfile",
            output / "projector.gguf",
            "--outtype",
            "f16",
        ],
        cancelled,
    )
    projectors = list(output.glob("*projector*.gguf"))
    if len(projectors) != 1:
        raise RuntimeError("Не найден однозначный экспортированный визуальный модуль")
    progress({"stage": "quantize"})
    command([quantizer, full, quantized, "Q4_K_M"], cancelled)
    progress({"stage": "cpu_evaluation", "pages": len(test)})
    cpu = (
        cpu_evaluate(quantized, projectors[0], test, payload, storage, cancelled) if test else None
    )
    files = [
        {"path": storage.relative(file), "sha256": digest(file), "size": file.stat().st_size}
        for file in (full, quantized, projectors[0])
    ]
    report = {
        "files": files,
        "merged": {"summary": aggregate(full_pages), "pages": full_pages},
        "q4_cpu": cpu,
        "quality_checked": bool(test and cpu and cpu["complete"]),
        "note": None
        if test
        else "Нет независимой тестовой выборки; качество экспорта не подтверждено",
    }
    atomic_json(output / "export.json", report)
    return report
