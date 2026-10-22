# Проверка сравнения OCR — 2026-10-22

Управляемое Linux-окружение, Python 3.12; CPU quota 2, RAM 8 GiB, без GPU.
Podman/mise здесь не установлены; контейнерная проверка — Docker/Compose.

Проверено:

- `pytest -q`: **43 passed, 11 skipped**. Пропущены явно включаемые тесты настоящих
  моделей (Paddle и 10 генеративных backend). Обычные тесты не измеряют OCR-качество.
- Ruff lint/format, `node --check`, `git diff --check`.
- `uv.lock`: разрешены CPU-зависимости; Nemotron отделён от KDL/Falcon из-за
  разных версий PyTorch/Transformers.
- Compose-конфигурация всех 11 профилей; у каждого собственные worker и models-init.
- Сборки targets `api`, `vlm`, `paddle`, `kdl`, `jina`, `falcon`, `nemotron`.
  Jina действительно компилирует отдельный бинарник с RoPE-поправкой.
- Nemotron: импорт torch 2.11.0+cpu / torchvision 0.26.0+cpu / timm 1.0.22 и
  AutoModel/AutoProcessor внутри контейнера `--network none`, CUDA недоступна.
- Paddle: импорт PaddlePaddle 3.2.2 и PPStructureV3 в контейнере без сети,
  `is_compiled_with_cuda() == False`.
- Falcon/KDL: PyTorch 2.7.1+cpu импортируется в контейнере без сети; KDL
  импортирует Qwen2VLForConditionalGeneration; Falcon импортирует CPU attention
  и requests. CUDA недоступна.
- API через Compose доступен с хоста на localhost:8000, клиент отдаётся;
  `/api/config` содержит 11 моделей. Загрузка трёхстраничного PDF в сравнение
  GLM/dots/Nemotron вернула HTTP 202 и три задания.
- GLM-сервис без весов пометил только собственное задание failed;
  dots/Nemotron сохранились queued. GLM остался работоспособен для следующих заданий.
  Затем сервис GLM остановлен, dots запущен отдельно.
- Тесты проверяют фильтрацию claim/recover по backend, миграцию прежней SQLite,
  атомарную публикацию сравнения и одинаковые байты PDF, продолжение после ошибки
  страницы, ограничение повторов, сохранение первого успешного прохода.
- Проверены контракты локального multimodal HTTP и Nemotron CPU processor/generate,
  адаптеры форматов, HTML/Markdown-таблицы, ведущие нули, пустые ячейки и OTSL.
  Falcon CPU attention сравнивается с PyTorch SDPA, включая image-prefix/decode masks;
  interleaved squared-ReLU gate проверяется отдельно.

Ограничения:

`huggingface.co`/хосты весов Hugging Face и официальный Paddle BOS заблокированы
сетевой политикой окружения (HTTPS CONNECT 403). Настоящие веса не загружены.
Полный inference, CPU-совместимость custom checkpoint-кода, качество русского
текста/таблиц, пиковая RAM и ток/с **не подтверждены**. Особенно важна проверка
CPU-порта Falcon и нового Transformers runtime Nemotron. Сборка/импорты и mock
контракты не заменяют прогон настоящих моделей.

Для реальной проверки подготовьте веса выбранного профиля и запустите opt-in:

```sh
OCR_RUN_MODEL_TESTS=1 OCR_REAL_BACKENDS=glm_ocr,surya \
  uv run --frozen pytest tests/test_real_vlm.py -v
```

Для Falcon нужны `--extra falcon`, для KDL `--extra kdl`, для Nemotron отдельное
окружение с `--extra nemotron`; для llama.cpp локально нужен соответствующий
CPU-бинарник (Jina — с поправкой). Проще запускать через готовые Compose targets.
Проверка является smoke test, для сравнения точности нужен размеченный набор.

Временный Compose-проект `ocr-compare-verify` и его тестовые тома удалены.
