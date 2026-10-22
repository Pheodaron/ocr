# VLM генерирует токены, но страница получает «пустой ответ»

Строки `n_decoded`, `eval time` и `stop processing` подтверждают генерацию,
но не показывают содержимое HTTP-ответа. Даже `truncated = 0` не гарантирует,
что `choices[0].message.content` содержит OCR-текст: это признак отсутствия
усечения входного контекста, а не проверка результата распознавания.

У LightOnOCR-3-0.8B используется
[шаблон Qwen с блоками thinking](https://huggingface.co/lightonai/LightOnOCR-3-0.8B/blob/main/chat_template.jinja).
Чат-парсер llama.cpp может выделять сгенерированный текст в `reasoning_content`
или `tool_calls`. Если адаптер читает только `content`, такой ответ будет
расценён как пустой. Без сохранённого исходного ответа это остаётся возможной
причиной конкретного сбоя, а не подтверждённым диагнозом.

Для всех OCR backend на llama.cpp приложение теперь запускает сервер с
`--skip-chat-parsing`, сохраняя Jinja-шаблон входа модели. Выход передаётся
как содержимое документа. Запрос дополнительно задаёт `reasoning_format=none`,
`parse_tool_calls=false`, `chat_template_kwargs.enable_thinking=false`.
Режим записывается в `engine.chat_output_parser` и `engine.enable_thinking`.
Разрешение страницы, веса и лимиты генерации от этих параметров не меняются.

Адаптер принимает строку или список текстовых частей `message.content`.
Для dots.mocr он также может восстановить явно оформленный layout JSON из
`reasoning_content`, добавив предупреждение для проверки. Произвольные
рассуждения модели результатом OCR не считаются.

Пустой ответ сопровождается диагностикой: `finish_reason`,
`completion_tokens`, имена полей сообщения и длина `reasoning_content`.
Исходный HTTP JSON сохраняется до разбора в
`/data/documents/<id>/page-<номер>-<dpi>.raw.json`, включая ошибочные страницы.
Журнал worker содержит строку `VLM response` с размерами полей, usage и timings;
сам текст документа в этой строке не печатается. Ошибка одной страницы
не останавливает следующие.

Обновление LightOnOCR-3-0.8B:

```sh
git pull --ff-only origin work
podman compose --profile lighton_small up --build --force-recreate -d
podman compose --profile lighton_small logs -f lighton-small
```

Затем загрузите PDF заново: завершённые ошибочные задания автоматически
не перезапускаются. Повторно скачивать подготовленные веса не требуется.
Остановка контейнеров сохраняет исходные ответы в data volume; удаление
задания через клиент удаляет и его файлы.

Описание флага и поведения парсера:
[llama.cpp b10000 server](https://github.com/ggml-org/llama.cpp/blob/47a39665e7081dc482feec169961acc09750a5c4/tools/server/README.md).
