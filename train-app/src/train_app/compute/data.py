"""Подготовка целей обучения с проверкой точной границы ответа."""

from pathlib import Path

from train_app.documents import model_image


def response_span(prefix_ids, full_ids, tokenizer, answer: str, max_tokens: int):
    """Проверяется без GPU; оба списка уже содержат развёрнутые визуальные токены."""
    boundary = len(prefix_ids)
    if full_ids[:boundary] != prefix_ids:
        raise ValueError("Шаблон обучения не совпал с префиксом генерации; обучение остановлено")
    if len(full_ids) > max_tokens:
        raise ValueError(
            f"Пример содержит {len(full_ids)} токенов при лимите {max_tokens}; обрезка запрещена"
        )
    try:
        end = full_ids.index(tokenizer.eos_token_id, boundary)
    except ValueError as error:
        raise ValueError("В целевом ответе отсутствует завершающий токен") from error
    decoded = tokenizer.decode(full_ids[boundary:end], skip_special_tokens=False)
    if decoded.strip() != answer.strip():
        raise ValueError("Декодированная обучающая цель не равна утверждённому ответу")
    if end == boundary and answer.strip():
        raise ValueError("Непустой ответ не содержит обучаемых токенов")
    return boundary, end


def training_example(
    native, image_path: Path, prompt: str, answer: str, longest_edge: int, max_tokens: int
):

    image = model_image(image_path, longest_edge)
    prefix = native.encode(image, prompt)
    full = native.encode(image, prompt, answer)
    prefix_ids = prefix["input_ids"][0]
    full_ids = full["input_ids"][0]
    boundary, end = response_span(
        prefix_ids.tolist(), full_ids.tolist(), native.processor.tokenizer, answer, max_tokens
    )
    labels = full_ids.clone()
    labels[:boundary] = -100
    labels[end + 1 :] = -100
    if int((labels != -100).sum()) <= 1 and answer.strip():
        raise ValueError("Непустой ответ не содержит обучаемых токенов")
    full["labels"] = labels.unsqueeze(0)
    return full
