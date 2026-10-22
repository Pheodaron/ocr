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


def test_local_multimodal_http_contract_and_cpu_server_lifecycle(settings, monkeypatch):
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
    engine = VLMEngine(settings, "glm_ocr")
    try:
        prediction = engine.recognize(np.zeros((20, 30, 3), dtype=np.uint8))
        assert prediction.format == "vlm"
        content = recorded["request"]["messages"][0]["content"]
        assert content[1]["text"] == "Text Recognition:"
        image_url = content[0]["image_url"]["url"]
        assert base64.b64decode(image_url.split(",")[1]).startswith(b"\x89PNG")
        assert recorded["request"]["temperature"] == 0
        command = recorded["command"]
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
