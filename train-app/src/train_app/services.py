"""HTTP-клиенты вычислительных служб. В этом модуле нет зависимостей CUDA."""

import asyncio
import base64
import io
import json
import time

import httpx

from .documents import model_image

QWEN_PROMPT = """Распознай весь видимый текст страницы в порядке чтения.
Верни только Markdown с HTML-таблицами. Сохрани rowspan и colspan, пустые ячейки,
знаки, десятичные запятые, ведущие нули, заголовки и примечания.
Не исправляй значения по смыслу. Не добавляй объяснений или ограждений кода."""


async def service_status(client: httpx.AsyncClient, name: str, url: str) -> dict:
    try:
        path = "/health" if name == "qwen" else "/status"
        response = await client.get(url + path, timeout=3)
        data = response.json()
        if name == "qwen":
            state = "ready" if response.status_code == 200 else "loading"
            slots = await client.get(url + "/slots", timeout=3)
            if slots.is_success and any(slot.get("is_processing") for slot in slots.json()):
                state = "busy"
            data = {"state": state, "runtime": "llama.cpp", "details": data}
        response.raise_for_status() if name != "qwen" else None
        return {"name": name, "reachable": True, **data, "checked_at": time.time()}
    except (httpx.HTTPError, ValueError, TypeError) as error:
        return {
            "name": name,
            "reachable": False,
            "state": "unreachable",
            "error": str(error),
            "checked_at": time.time(),
        }


async def all_statuses(settings) -> list[dict]:
    async with httpx.AsyncClient(trust_env=False) as client:
        return await asyncio.gather(
            *(
                service_status(client, name, url)
                for name, url in (
                    ("lighton4b", settings.lighton_url),
                    ("qwen", settings.qwen_url),
                    ("trainer", settings.trainer_url),
                )
            )
        )


async def qwen_idle(client, url: str) -> bool:
    response = await client.get(url + "/slots", timeout=5)
    response.raise_for_status()
    slots = response.json()
    return isinstance(slots, list) and len(slots) == 1 and not slots[0].get("is_processing", True)


async def qwen_recognize(client, settings, image_path, cancelled, progress) -> dict:
    """Поток нужен для отмены и диагностики; завершённый ответ публикуется целиком."""
    image = await asyncio.to_thread(model_image, image_path, settings.model_longest_edge)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    uri = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
    body = {
        "model": settings.qwen_model,
        "temperature": 0,
        "seed": 42,
        "max_tokens": settings.max_output_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": QWEN_PROMPT},
                    {"type": "image_url", "image_url": {"url": uri}},
                ],
            }
        ],
    }
    chunks, content, reason, usage = [], [], None, {}
    async with client.stream(
        "POST",
        settings.qwen_url + "/v1/chat/completions",
        json=body,
        timeout=settings.inference_timeout,
    ) as response:
        response.raise_for_status()
        async for line in response.aiter_lines():
            if cancelled():
                raise asyncio.CancelledError("Операция отменена пользователем")
            if not line.startswith("data: "):
                continue
            if line == "data: [DONE]":
                break
            chunk = json.loads(line[6:])
            chunks.append(chunk)
            if chunk.get("usage"):
                usage = chunk["usage"]
            for choice in chunk.get("choices", []):
                delta = choice.get("delta", {}).get("content")
                if isinstance(delta, str):
                    content.append(delta)
                if choice.get("finish_reason"):
                    reason = choice["finish_reason"]
            progress({"stage": "generation", "output_characters": sum(map(len, content))})
    raw = {
        "chunks": chunks,
        "usage": usage,
        "finish_reason": reason,
        "input_size": image.size,
        "prompt": QWEN_PROMPT,
        "thinking": False,
    }
    text = "".join(content).strip()
    if reason == "length":
        return {"raw": raw, "error": "Ответ обрезан по лимиту токенов", "text": text}
    if reason not in {"stop", "eos"}:
        return {
            "raw": raw,
            "error": "Поток завершился без подтверждения конца ответа",
            "text": text,
        }
    if not text:
        return {"raw": raw, "error": "Модель вернула пустой содержательный ответ", "text": ""}
    return {"raw": raw, "text": text, "metadata": {"usage": usage, "input_size": image.size}}
