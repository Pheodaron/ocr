"""One local llama.cpp multimodal server per document; no remote inference."""

import base64
import ctypes
import http.client
import json
import logging
import math
import os
import signal
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

import pymupdf

from .config import Settings
from .engine import InferenceProcessError, Prediction
from .vlm_models import model_directory, verify_vlm

LOGGER = logging.getLogger(__name__)


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
        "auto" if backend == "dots_mocr" else "off",
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


def encode_image(image, max_pixels: int) -> tuple[bytes, dict]:
    height, width = image.shape[:2]
    scale = min(1.0, math.sqrt(max_pixels / (width * height)))
    target_width, target_height = max(1, int(width * scale)), max(1, int(height * scale))
    if target_width * target_height > max_pixels:
        # Clamping a subpixel dimension to one can otherwise exceed the area budget.
        if target_width >= target_height:
            target_width = max(1, max_pixels // target_height)
        else:
            target_height = max(1, max_pixels // target_width)
    pixmap = pymupdf.Pixmap(pymupdf.csRGB, width, height, image[:, :, ::-1].tobytes(), False)
    if (target_width, target_height) != (width, height):
        pixmap = pymupdf.Pixmap(pixmap, target_width, target_height)
    preprocessing = {
        "render_width": width,
        "render_height": height,
        "input_width": pixmap.width,
        "input_height": pixmap.height,
        "max_pixels": max_pixels,
        "resized": scale < 1.0,
    }
    return pixmap.tobytes("png"), preprocessing


class VLMEngine:
    retry_on_review = False

    def __init__(self, settings: Settings, backend: str):
        self.settings = settings
        self.backend = backend
        self.process = None
        self.log = None
        manifest = verify_vlm(backend, settings)
        self.manifest = manifest
        self.server_restarts = 0
        self.startup_error = None
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
            "max_input_pixels": settings.vlm_image_budget(backend),
            "cache_type_k": settings.kv_cache_type,
            "cache_type_v": "f16",
            "flash_attention": "auto" if backend == "dots_mocr" else "off",
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
            self._start_server()
        except BaseException:
            self.close()
            raise

    def _start_server(self):
        self.log_offset = self.log.tell()
        try:
            # An occupied endpoint must never silently become a different OCR model.
            import socket

            with socket.socket() as probe:
                probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                probe.bind(("127.0.0.1", self.settings.llama_port))
            self.process = subprocess.Popen(
                server_command(self.settings, self.backend, self.manifest),
                stdout=self.log,
                stderr=subprocess.STDOUT,
                preexec_fn=parent_death_signal,
            )
            deadline = time.monotonic() + self.settings.vlm_startup_seconds
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    raise self._process_failure()
                try:
                    with self.http.open(self.endpoint + "/health", timeout=1) as response:
                        if response.status == 200:
                            return
                except (urllib.error.URLError, TimeoutError):
                    pass
                time.sleep(0.2)
            raise TimeoutError(f"Таймаут загрузки {self.backend}; см. {self.log.name}")
        except BaseException:
            self._stop_process()
            raise

    def _process_failure(self):
        code = self.process.poll() if self.process is not None else None
        status = f"код {code}"
        if code is not None and code < 0:
            try:
                status += f" ({signal.Signals(-code).name})"
            except ValueError:
                pass
        detail = ""
        if self.log is not None:
            # Inspect only this server instance, not an earlier document in the same log.
            try:
                with Path(self.log.name).open("rb") as source:
                    size = source.seek(0, os.SEEK_END)
                    source.seek(max(self.log_offset, size - 8192))
                    lines = source.read().decode("utf-8", errors="replace").splitlines()
            except OSError:
                lines = []
            failures = [
                line
                for line in lines
                if any(
                    word in line.lower()
                    for word in (
                        "error",
                        "failed",
                        "assert",
                        "out of memory",
                        "fatal",
                        "unsupported",
                    )
                )
            ]
            detail = " | ".join((failures or lines[-3:])[-3:])[-700:]
        return InferenceProcessError(
            f"llama-server завершился: {status}; журнал {self.log.name}. {detail}"
        )

    def _ensure_server(self):
        if self.startup_error:
            raise InferenceProcessError(self.startup_error)
        if self.process is None or self.process.poll() is not None:
            LOGGER.warning(
                "Restarting llama-server before the next page (%s): %s",
                self.backend,
                self._process_failure() if self.process is not None else "transport interrupted",
            )
            self.server_restarts += 1
            try:
                self._start_server()
            except Exception as error:
                self.startup_error = f"Не удалось перезапустить llama-server: {error}"
                raise InferenceProcessError(self.startup_error) from error

    def _version(self):
        result = subprocess.run(
            [self.settings.llama_server, "--version"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        return (result.stdout + result.stderr).strip()[:1000]

    def _stop_process(self):
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        self.process = None

    def close(self):
        self._stop_process()
        if self.log is not None:
            self.log.close()

    def recognize(self, image, **_kwargs) -> Prediction:
        if self.log.closed:
            raise InferenceProcessError("Движок OCR закрыт")
        self._ensure_server()
        png, preprocessing = encode_image(image, self.settings.vlm_image_budget(self.backend))
        if preprocessing["resized"]:
            LOGGER.info("VLM image resized: %s", preprocessing)
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
            if self.process.poll() is not None:
                failure = self._process_failure()
                self._stop_process()
                raise failure from error
            raise RuntimeError(f"llama-server HTTP {error.code}: {detail}") from error
        except (OSError, http.client.HTTPException) as error:
            if self.process.poll() is not None:
                failure = self._process_failure()
            else:
                failure = InferenceProcessError(
                    f"Сбой запроса llama-server: {error}; журнал {self.log.name}. "
                    "Процесс остановлен; перед следующей страницей будет перезапущен."
                )
            self._stop_process()
            raise failure from error
        raw.update(
            ocr_backend=self.backend,
            ocr_task=self.settings.glm_task,
            image_preprocessing=preprocessing,
        )
        return Prediction(
            raw=raw,
            image=image,
            format="vlm",
            metrics={"server_restarts": self.server_restarts, "image_preprocessing": preprocessing},
        )
