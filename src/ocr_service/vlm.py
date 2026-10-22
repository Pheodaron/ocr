"""One local llama.cpp multimodal server per document; no remote inference."""

import base64
import ctypes
import json
import os
import signal
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

import pymupdf

from .config import Settings
from .engine import Prediction
from .vlm_models import model_directory, verify_vlm


def parent_death_signal():
    # Linux worker runs in a separate process. Kill its server on a native crash/OOM too.
    parent = os.getppid()
    ctypes.CDLL(None).prctl(1, signal.SIGTERM)
    if os.getppid() != parent:
        os.kill(os.getpid(), signal.SIGTERM)


def server_command(settings: Settings, backend: str, manifest: dict) -> list[str]:
    directory = model_directory(backend, settings).resolve()
    context = context_size(settings, backend)
    return [
        settings.llama_server,
        "--model",
        str(directory / manifest["model"]),
        "--mmproj",
        str(directory / manifest["projector"]),
        "--alias",
        backend,
        "--host",
        "127.0.0.1",
        "--port",
        str(settings.llama_port),
        "--n-gpu-layers",
        "0",
        "--no-mmproj-offload",
        "--threads",
        str(settings.cpu_threads),
        "--threads-batch",
        str(settings.cpu_threads),
        "--ctx-size",
        str(context),
        "--parallel",
        "1",
        "--cache-type-k",
        settings.kv_cache_type,
        "--cache-type-v",
        "f16",
        "--flash-attn",
        "off",
        "--fit",
        "off",
        "--jinja",
        "--offline",
    ]


def context_size(settings: Settings, backend: str) -> int:
    return (
        settings.dots_context
        if backend == "dots_mocr"
        else (settings.glm_context if backend == "glm_ocr" else settings.vlm_context)
    )


class VLMEngine:
    retry_on_review = False

    def __init__(self, settings: Settings, backend: str):
        self.settings = settings
        self.backend = backend
        self.process = None
        self.log = None
        manifest = verify_vlm(backend, settings)
        self.prompt = {
            "dots_mocr": lambda: Path(__file__).with_name("dots_prompt.txt").read_text(),
            "glm_ocr": lambda: (
                "Table Recognition:" if settings.glm_task == "table" else "Text Recognition:"
            ),
            "lighton": lambda: "grounding",
            "lighton_small": lambda: "grounding",
            "jina": lambda: (
                "Transcribe the provided document image into a clean Markdown format, "
                "preserving the natural reading order."
            ),
            "surya": lambda: (
                "OCR this image to HTML. Each block is a div with data-label and data-bbox "
                "(x0 y0 x1 y1, normalized 0-1000)."
            ),
            "chandra": lambda: Path(__file__).with_name("chandra_prompt.txt").read_text(),
        }[backend]()
        self.metadata = {
            "name": backend,
            "runtime": "llama.cpp",
            "runtime_version": self._version(),
            "runtime_patch": os.environ.get("OCR_LLAMA_PATCH"),
            "manifest": manifest,
            "prompt": self.prompt,
            "temperature": 0,
            "seed": 42,
            "cpu_threads": settings.cpu_threads,
            "context": context_size(settings, backend),
            "cache_type_k": settings.kv_cache_type,
            "cache_type_v": "f16",
            "task": settings.glm_task if backend == "glm_ocr" else "layout",
            "coordinates": "not normalized; generated dots boxes are retained only in raw output",
        }
        # Never send loopback inference through an HTTP proxy inherited by the worker.
        self.http = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        self.endpoint = f"http://127.0.0.1:{settings.llama_port}"
        logs = settings.data_dir / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        self.log = (logs / f"llama-{backend}.log").open("ab")
        try:
            # An occupied endpoint must never silently become a different OCR model.
            import socket

            with socket.socket() as probe:
                probe.bind(("127.0.0.1", settings.llama_port))
            self.process = subprocess.Popen(
                server_command(settings, backend, manifest),
                stdout=self.log,
                stderr=subprocess.STDOUT,
                preexec_fn=parent_death_signal,
            )
            deadline = time.monotonic() + settings.vlm_startup_seconds
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    raise RuntimeError(f"llama-server завершился; см. {self.log.name}")
                try:
                    with self.http.open(self.endpoint + "/health", timeout=1) as response:
                        if response.status == 200:
                            return
                except (urllib.error.URLError, TimeoutError):
                    pass
                time.sleep(0.2)
            raise TimeoutError(f"Таймаут загрузки {backend}; см. {self.log.name}")
        except BaseException:
            self.close()
            raise

    def _version(self):
        result = subprocess.run(
            [self.settings.llama_server, "--version"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        return (result.stdout + result.stderr).strip()[:1000]

    def close(self):
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        if self.log is not None:
            self.log.close()

    def recognize(self, image, **_kwargs) -> Prediction:
        if self.process.poll() is not None:
            raise RuntimeError("llama-server остановлен; проверьте лимит RAM и журнал")
        height, width = image.shape[:2]
        png = pymupdf.Pixmap(
            pymupdf.csRGB, width, height, image[:, :, ::-1].tobytes(), False
        ).tobytes("png")
        payload = {
            "model": self.backend,
            "temperature": 0,
            "seed": 42,
            "max_tokens": self.settings.vlm_max_tokens,
            "stream": False,
            "cache_prompt": False,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/png;base64,"
                                + base64.b64encode(png).decode("ascii")
                            },
                        },
                        {"type": "text", "text": self.prompt},
                    ],
                }
            ],
        }
        request = urllib.request.Request(
            self.endpoint + "/v1/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with self.http.open(request, timeout=self.settings.vlm_timeout_seconds) as response:
                body = response.read(32 * 1024**2 + 1)
            if len(body) > 32 * 1024**2:
                raise ValueError("Слишком большой ответ VLM")
            raw = json.loads(body)
        except urllib.error.HTTPError as error:
            detail = error.read(2000).decode("utf-8", errors="replace")
            raise RuntimeError(f"llama-server HTTP {error.code}: {detail}") from error
        raw.update(ocr_backend=self.backend, ocr_task=self.settings.glm_task)
        return Prediction(raw=raw, image=image, format="vlm")
