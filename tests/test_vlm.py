import base64
import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import Mock

import numpy as np
import pytest

from ocr_service.engine import Prediction
from ocr_service.processing import process_job
from ocr_service.vlm import VLMEngine, server_command
from ocr_service.vlm_adapter import adapt_vlm


def response(backend, content, finish="stop"):
    return {
        "ocr_backend": backend,
        "choices": [{"message": {"content": content}, "finish_reason": finish}],
        "usage": {"completion_tokens": 20},
        "timings": {"predicted_n": 20, "predicted_ms": 4000},
    }


@pytest.mark.parametrize(
    "backend,content",
    [
        (
            "glm_ocr",
            '<table><tr><td>00125</td><td rowspan="2">150 кВт</td></tr><tr><td></td></tr></table>',
        ),
        (
            "dots_mocr",
            json.dumps(
                [
                    {
                        "category": "Table",
                        "bbox": [1, 2, 3, 4],
                        "text": "<table><tr><td>00125</td><td>150 кВт</td></tr></table>",
                    }
                ]
            ),
        ),
        (
            "chandra",
            '<div data-bbox="0 0 1000 1000" data-label="Table">'
            "<table><tr><td>00125</td><td>150 кВт</td></tr></table></div>",
        ),
        (
            "lighton",
            "![table](0,0,1000,1000)\n<table><tr><td>00125</td><td>150 кВт</td></tr></table>",
        ),
    ],
)
def test_model_formats_preserve_table_strings_without_fabricating_scores(backend, content):
    page = adapt_vlm(response(backend, content), 1, 250, 1000, 1500)
    assert page.tables[0].cells[0].text == "00125"
    assert page.tables[0].cells[1].text == "150 кВт"
    assert page.confidence is None and page.needs_review
    assert page.metrics["usage"]["completion_tokens"] == 20
    if backend in {"lighton", "chandra"}:
        assert page.blocks[0].bbox == [0, 0, 1000, 1500]
    if backend == "dots_mocr":
        assert page.blocks[0].bbox is None


def test_truncation_is_visible_and_empty_response_is_an_error():
    page = adapt_vlm(response("dots_mocr", '[{"text": "incomplete', "length"), 1, 250, 100, 100)
    assert "incomplete" in page.text
    assert any("length" in warning for warning in page.warnings)
    assert page.needs_review and not page.tables
    with pytest.raises(ValueError, match="пустой"):
        adapt_vlm(response("glm_ocr", ""), 1, 250, 100, 100)


def test_generative_review_does_not_trigger_unnecessary_dpi_retry(client, pdf_bytes, settings):
    job = client.post(
        "/api/documents", files={"file": ("test.pdf", pdf_bytes)}, data={"backend": "glm_ocr"}
    ).json()
    db = client.app.state.db
    engine = Mock(metadata={"name": "contract-fixture"}, retry_on_review=False)
    engine.recognize.side_effect = lambda image, **kwargs: Prediction(
        raw=response("glm_ocr", "Договор 00125"), image=image, format="vlm"
    )
    process_job(db, db.claim_job(), engine, settings)
    assert engine.recognize.call_count == 3
    assert all(len(page.passes) == 1 and page.needs_review for page in db.pages(job["id"]))


@pytest.mark.parametrize("backend", ["glm_ocr", "dots_mocr", "lighton", "lighton_small"])
def test_local_multimodal_http_contract_and_cpu_server_lifecycle(settings, monkeypatch, backend):
    import pymupdf

    from ocr_service import vlm

    recorded = {}
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    settings = settings.model_copy(update={"llama_port": port})

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"status":"ok"}')

        def do_POST(self):
            recorded["request"] = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps(response("glm_ocr", "Договор 00125")).encode())

    class Process:
        def __init__(self, command, **_kwargs):
            recorded["command"] = command
            self.server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
            self.thread = threading.Thread(target=self.server.serve_forever)
            self.thread.start()
            self.stopped = False

        def poll(self):
            return 0 if self.stopped else None

        def terminate(self):
            self.server.shutdown()
            self.server.server_close()
            self.thread.join()
            self.stopped = True
            recorded["closed"] = True

        def wait(self, **_kwargs):
            return 0

    manifest = {"model": "model.gguf", "projector": "projector.gguf"}
    monkeypatch.setattr(vlm, "verify_vlm", lambda *_args: manifest)
    monkeypatch.setattr(VLMEngine, "_version", lambda _self: "test-fixture")
    monkeypatch.setattr(vlm.subprocess, "Popen", Process)
    engine = VLMEngine(settings, backend)
    try:
        prediction = engine.recognize(np.zeros((2400, 1800, 3), dtype=np.uint8))
        assert prediction.format == "vlm"
        content = recorded["request"]["messages"][0]["content"]
        assert content[1]["text"] == engine.prompt
        assert recorded["request"]["model"] == backend
        image_url = content[0]["image_url"]["url"]
        decoded = pymupdf.Pixmap(base64.b64decode(image_url.split(",")[1]))
        expected_budget = 2_500_000
        assert decoded.width * decoded.height <= expected_budget
        assert decoded.width * decoded.height >= expected_budget * 0.99
        assert engine.metadata["max_input_pixels"] == expected_budget
        assert recorded["request"]["temperature"] == 0
        assert engine.metadata["chat_output_parser"] == "content"
        assert recorded["request"]["reasoning_format"] == "none"
        assert recorded["request"]["parse_tool_calls"] is False
        assert recorded["request"]["chat_template_kwargs"]["enable_thinking"] is False
        assert "--skip-chat-parsing" in recorded["command"]
        command = recorded["command"]
        expected_flash_attention = "auto" if backend == "dots_mocr" else "off"
        assert command[command.index("--flash-attn") + 1] == expected_flash_attention
        assert engine.metadata["flash_attention"] == expected_flash_attention
        assert command[command.index("--n-gpu-layers") + 1] == "0"
        assert "--no-mmproj-offload" in command and "--offline" in command
    finally:
        engine.close()
    assert recorded["closed"]
    assert (
        server_command(settings.model_copy(update={"glm_context": 131072}), "glm_ocr", manifest)[
            command.index("--ctx-size") + 1
        ]
        == "131072"
    )


def test_server_crash_skips_higher_dpi_and_next_page_restarts_server(
    client, pdf_bytes, settings, monkeypatch
):
    import os

    from ocr_service import vlm

    job = client.post(
        "/api/documents", files={"file": ("crash.pdf", pdf_bytes)}, data={"backend": "glm_ocr"}
    ).json()
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    settings = settings.model_copy(update={"llama_port": port})
    processes, requests = [], []
    logs = settings.data_dir / "logs"
    logs.mkdir(parents=True)
    (logs / "llama-glm_ocr.log").write_text("fatal: unrelated previous document\n")

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"status":"ok"}')

        def do_POST(self):
            requests.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            if len(requests) == 2:
                process = processes[-1]
                os.write(process.log.fileno(), b"GGML_ASSERT fixture: native page crash\n")
                process.exit_code = -6
                process.stop_listener()
                self.close_connection = True
                return
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps(response("glm_ocr", "Договор 00125")).encode())

    class Process:
        def __init__(self, command, *, stdout, **kwargs):
            self.log = stdout
            self.exit_code = None
            self.server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
            self.server.daemon_threads = True
            self.thread = threading.Thread(target=self.server.serve_forever)
            processes.append(self)
            self.thread.start()

        def poll(self):
            return self.exit_code

        def stop_listener(self):
            self.server.shutdown()
            self.server.server_close()
            self.thread.join()

        def terminate(self):
            self.exit_code = 0
            self.stop_listener()

        def wait(self, **kwargs):
            return self.exit_code

    monkeypatch.setattr(vlm, "verify_vlm", lambda *_: {"model": "m.gguf", "projector": "v.gguf"})
    monkeypatch.setattr(VLMEngine, "_version", lambda _: "crash-contract-fixture")
    monkeypatch.setattr(vlm.subprocess, "Popen", Process)
    engine = VLMEngine(settings, "glm_ocr")
    try:
        db = client.app.state.db
        process_job(db, db.claim_job(), engine, settings)
        pages = db.pages(job["id"])
        assert [page.status for page in pages] == ["completed", "failed", "completed"]
        assert db.summary(job["id"]).status == "partial"
        assert len(requests) == 3  # no 350-DPI request to the failed server
        assert len(processes) == 2
        assert "SIGABRT" in pages[1].error and "GGML_ASSERT fixture" in pages[1].error
        assert "unrelated previous document" not in pages[1].error
        assert pages[2].metrics["server_restarts"] == 1
    finally:
        engine.close()


def test_failed_server_restart_does_not_loop_on_every_remaining_page():
    from ocr_service.engine import InferenceProcessError

    engine = object.__new__(VLMEngine)
    engine.backend = "glm_ocr"
    engine.log = Mock(closed=False)
    engine.process = None
    engine.startup_error = None
    engine.server_restarts = 0
    engine._start_server = Mock(side_effect=RuntimeError("fixture startup failed"))
    for _ in range(3):
        with pytest.raises(InferenceProcessError, match="fixture startup failed"):
            engine.recognize(np.zeros((10, 10, 3), dtype=np.uint8))
    engine._start_server.assert_called_once()


def test_vlm_image_budget_preserves_aspect_ratio_colors_and_original_preview():
    import pymupdf

    from ocr_service.vlm import encode_image

    source = np.zeros((1500, 2000, 3), dtype=np.uint8)
    source[:, :, 0] = 255  # blue in BGR
    png, metadata = encode_image(source, 1_500_000)
    decoded = pymupdf.Pixmap(png)
    assert decoded.width * decoded.height <= 1_500_000
    assert abs(decoded.width / decoded.height - 2000 / 1500) < 0.002
    assert decoded.samples[:3] == bytes([0, 0, 255])
    assert source.shape == (1500, 2000, 3)
    assert metadata["render_width"] == 2000 and metadata["resized"]
    small = source[:20, :30]
    png, metadata = encode_image(small, 1_500_000)
    assert (pymupdf.Pixmap(png).width, pymupdf.Pixmap(png).height) == (30, 20)
    assert not metadata["resized"]


@pytest.mark.parametrize("backend", ["dots_mocr", "glm_ocr"])
def test_standard_a4_keeps_full_150_dpi_resolution(settings, tmp_path, backend):
    import pymupdf

    from ocr_service.processing import render_page
    from ocr_service.vlm import encode_image

    source = tmp_path / "a4.pdf"
    with pymupdf.open() as document:
        page = document.new_page(width=210 / 25.4 * 72, height=297 / 25.4 * 72)
        page.insert_text((40, 60), "Document with 11 pt text", fontsize=11)
        document.save(source)
    image = render_page(source, 1, settings.vlm_dpi, settings.max_page_pixels)
    png, metadata = encode_image(image, settings.vlm_image_budget(backend))
    decoded = pymupdf.Pixmap(png)
    assert (decoded.width, decoded.height) == (1241, 1754)
    assert not metadata["resized"]


def test_vlm_resize_is_visible_and_bboxes_still_refer_to_original_preview():
    raw = response("lighton", "![text](0,0,1000,1000)\nМелкий текст")
    raw["image_preprocessing"] = {
        "resized": True,
        "render_width": 2000,
        "render_height": 3000,
        "input_width": 1000,
        "input_height": 1500,
    }
    page = adapt_vlm(raw, 1, 250, 2000, 3000)
    assert page.blocks[0].bbox == [0, 0, 2000, 3000]
    assert any("1000×1500" in warning for warning in page.warnings)


def test_dots_recovers_only_explicit_ocr_layout_from_reasoning_field():
    raw = response("dots_mocr", None)
    raw["choices"][0]["message"]["reasoning_content"] = json.dumps(
        [{"category": "Table", "text": "<table><tr><td>00125</td></tr></table>"}]
    )
    page = adapt_vlm(raw, 1, 150, 100, 100)
    assert page.tables[0].cells[0].text == "00125"
    assert page.needs_review
    assert any("reasoning_content" in warning for warning in page.warnings)
    assert raw["choices"][0]["message"]["content"] is None  # preserve the evidence


@pytest.mark.parametrize(
    "backend,reasoning",
    [
        ("dots_mocr", "Let me inspect the document. The contract might be 00125."),
        ("dots_mocr", '[{"category": "Text", "text": "truncated'),
        ("dots_mocr", '{"analysis": "00125"}'),
        ("glm_ocr", '[{"category": "Text", "text": "00125"}]'),
    ],
)
def test_reasoning_is_not_silently_presented_as_ocr(backend, reasoning):
    raw = response(backend, "")
    raw["choices"][0]["message"]["reasoning_content"] = reasoning
    with pytest.raises(ValueError, match="completion_tokens=20") as error:
        adapt_vlm(raw, 2, 150, 100, 100)
    assert f"reasoning_chars={len(reasoning)}" in str(error.value)
    assert "page-2-150.raw.json" in str(error.value)
    assert reasoning not in str(error.value)


def test_text_parts_and_final_content_take_priority_over_reasoning():
    raw = response("dots_mocr", [{"type": "text", "text": '[{"category":"Text","text":"00125"}]'}])
    raw["choices"][0]["message"]["reasoning_content"] = "This must not appear in OCR"
    page = adapt_vlm(raw, 1, 150, 100, 100)
    assert page.text == "00125"
    assert not any("reasoning_content" in warning for warning in page.warnings)


@pytest.mark.parametrize("backend", ["dots_mocr", "lighton_small"])
def test_empty_vlm_response_is_saved_and_does_not_block_following_pages(
    client, pdf_bytes, settings, backend
):
    from ocr_service.processing import document_path

    job = client.post(
        "/api/documents", files={"file": ("test.pdf", pdf_bytes)}, data={"backend": backend}
    ).json()
    answers = [response(backend, None), response(backend, ""), response(backend, "")]
    answers[0]["choices"][0]["message"]["reasoning_content"] = "Not an OCR result"
    layout = json.dumps([{"category": "Text", "text": "Договор 00125"}])
    if backend == "dots_mocr":
        answers[1]["choices"][0]["message"]["reasoning_content"] = layout
        answers[2]["choices"][0]["message"]["content"] = layout
    else:
        grounded = "![text](0,0,1000,1000)\nДоговор 00125"
        answers[1]["choices"][0]["message"]["content"] = grounded
        answers[2]["choices"][0]["message"]["content"] = grounded
    engine = Mock(metadata={"name": "contract-fixture"}, retry_on_review=False)
    pending = iter(answers)
    engine.recognize.side_effect = lambda image, **kwargs: Prediction(
        raw=next(pending), image=image, format="vlm"
    )
    db = client.app.state.db
    process_job(db, db.claim_job(), engine, settings)
    pages = db.pages(job["id"])
    assert [page.status for page in pages] == ["failed", "completed", "completed"]
    assert pages[1].text == pages[2].text
    assert "Договор 00125" in pages[1].text
    if backend == "lighton_small":
        assert pages[1].blocks[0].text == "Договор 00125"
    assert engine.recognize.call_count == 3
    saved = document_path(settings, job["id"]) / "page-1-150.raw.json"
    assert json.loads(saved.read_text()) == answers[0]
    assert db.summary(job["id"]).status == "partial"
