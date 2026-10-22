# PDF OCR на CPU

Локальный Linux-сервис: загрузка PDF → асинхронное распознавание текста и таблиц →
JSON, Markdown и HTML. Русский и английский. Веб-клиент включён в приложение.

**[Пошаговый гайд по развёртыванию](docs/deployment.md)**: Linux/Podman/mise,
11 моделей, переключение сервисов, API, резервные копии и перенос без интернета.

## Запуск через Podman Compose

Требуются Linux x86_64, CPU с AVX2, Podman и Compose provider (`podman-compose`
либо Docker Compose, вызываемый через `podman compose`). Для сборки и первой
подготовки весов нужен интернет. Полный набор из 11 вариантов занимает десятки ГБ вместе с образами.
На машине с ограниченной RAM запускайте один профиль за раз;
потребление RAM и скорость нужно измерить на ваших документах.

```sh
# Общий API и только GLM-OCR; остальные модели не запускаются и не скачиваются.
podman compose --profile glm_ocr up --build -d
podman compose logs -f models-glm-ocr glm-ocr
```

Клиент: **http://localhost:8000**. Справка: **http://localhost:8000/docs**.
Обычный `podman compose up -d` без профилей запускает только API.

| Модель | Compose-профиль | Сервис |
| --- | --- | --- |
| Paddle | `paddle` | `paddle` |
| dots.mocr | `dots_mocr` | `dots-mocr` |
| GLM-OCR | `glm_ocr` | `glm-ocr` |
| Chandra OCR 2 | `chandra` | `chandra` |
| LightOnOCR-3-1B | `lighton` | `lighton` |
| KDL nano | `kdl` | `kdl` |
| LightOnOCR-3-0.8B | `lighton_small` | `lighton-small` |
| Jina OCR v1 | `jina` | `jina` |
| Surya OCR 2 | `surya` | `surya` |
| Falcon-OCR 1.5 | `falcon` | `falcon` |
| Nemotron 3.5 Parse | `nemotron` | `nemotron` |

У каждого профиля свой `models-*` для подготовки только его весов. Runtime-сервисы
не скачивают модели. Образы API, VLM, Paddle, KDL, Jina, Falcon и Nemotron разделены; VLM не устанавливает
Paddle/PyTorch. У всех сервисов общие данные и общий каталог весов, но разные
очереди заданий и блокировки по backend.

Чтобы сравнить на машине с ограниченной RAM, отметьте несколько конфигураций в
клиенте и загрузите PDF, затем запускайте сервисы по одному:

```sh
# Дождитесь окончания прогона GLM в клиенте, затем:
podman compose stop glm-ocr
podman compose --profile dots_mocr up --build -d
# После прогона dots:
podman compose stop dots-mocr
podman compose --profile lighton up --build -d
```

Задания выключенных моделей остаются в очереди с отметкой «Ожидает запуска
сервиса». Уже готовые результаты видны в общей таблице сравнения.
Одновременный запуск нескольких профилей разрешён только по вашему выбору:
тогда потребление RAM и CPU складывается. Приложение не включает их автоматически.
`OCR_ENABLED_BACKENDS` ограничивает список вариантов в API/клиенте; запуском
контейнеров управляют Compose-профили.

Повторные запуски используют сохранённые модели. Для настройки скопируйте
`.env.example` в `.env`. Например, `OCR_CPU_THREADS=2` и `OCR_PORT=8080`.

```sh
podman compose down
```

Остановка сохраняет именованные тома с моделями и данными. `down -v` удаляет
эти тома вместе с результатами. В этой версии автоматической очистки нет.

## Что запускается

- `api`: FastAPI/Uvicorn на localhost; веб-страница без CDN и без весов.
- `paddle`, `dots-mocr`, `glm-ocr`, `chandra`, `lighton`, `kdl`, `lighton-small`,
  `jina`, `surya`, `falcon`, `nemotron`: независимые
  OCR-сервисы в своих профилях. У каждого `network_mode: none`, свой heartbeat,
  свой backend-фильтр очереди и один распознаваемый документ одновременно.
- `models-*`: одноразовая подготовка весов соответствующего сервиса.
- Постоянные тома `data` и `models`. SQLite размещается на локальном Linux-диске,
  а не на NFS/SMB.

Каждый сервис запускает документ в отдельном процессе; после документа процесс
завершается, освобождая память модели. Другие сервисы не обязаны быть запущены.
Время загрузки включает проверку весов и учитывается отдельно от распознавания.
Ошибка страницы не останавливает следующие страницы. После перезапуска сервиса
завершённые страницы сохраняются; незавершённая повторяется. После двух
прерываний на одной странице она отмечается ошибочной, и документ идёт дальше.
Перезапуск сервиса восстанавливает только задания его backend.

SIGTERM запрашивает остановку после текущей страницы. Compose даёт на это
120 секунд; при принудительном завершении восстановление работает тем же образом.

## API

```sh
curl -F 'file=@document.pdf' -F 'backend=glm_ocr' http://localhost:8000/api/documents
# Ответ 202: id, status=queued, page_count и счётчики прогресса.

curl http://localhost:8000/api/documents/ID
curl http://localhost:8000/api/documents/ID/result
curl -o result.md 'http://localhost:8000/api/documents/ID/download?format=markdown'
```

| Endpoint | Описание |
| --- | --- |
| `POST /api/documents` | multipart/form-data: `file`, `backend` (по умолчанию `paddle`) |
| `POST /api/comparisons` | `file` и повторяемое поле `backends`; один PDF, несколько движков |
| `GET /api/comparisons/{id}` | Прогоны сравнения и их статусы/время |
| `GET /api/config` | Доступные конфигурации и ограничения |
| `GET /api/documents` | Последние 30 документов |
| `GET /api/documents/{id}` | Статус и счётчики страниц |
| `GET /api/documents/{id}/result` | Результат; 409, пока документ не завершён |
| `GET /api/documents/{id}/download?format=json\|markdown\|html` | Скачать результат |
| `GET /api/documents/{id}/pages/{number}/image` | Сохранённый PNG, нумерация с 1 |
| `GET /health` | API/SQLite и состояние worker |
| `GET /ready` | 200 при готовом worker, иначе 503 |

Терминальные статусы: `completed` — все страницы обработаны; `partial` — есть
успешные и ошибочные; `failed` — ни одной успешной. Ошибки отдельных страниц
находятся в `pages[].error`. Признак `needs_review` независим от статуса:
страница может быть распознана, но нуждаться в проверке.

JSON имеет `schema_version: "1.0"`, SHA256 исходного PDF, сведения о моделях и
настройках, текст, блоки, OCR-фрагменты, таблицы и предупреждения каждой страницы.
`bbox = [x_min, y_min, x_max, y_max]` в пикселях сохранённого изображения после
исправления ориентации; вместе с ним возвращаются DPI, размеры и угол.
`block.bbox` и `table.bbox` могут быть `null`, если надёжного сопоставления с
изображением нет. Для dots координаты остаются в raw-ответе: llama.cpp может
изменять размер изображения внутри vision encoder.
Ячейки содержат строки, индексы с нуля и `rowspan`/`colspan`; индивидуальные
оценки OCR не выдаются за уверенность структуры или каждой HTML-ячейки.

## Конфигурации для сравнения

| ID | Модель | Веса | Runtime / маршрут |
| --- | --- | --- | --- |
| `paddle` | PP-StructureV3 | Исходные Paddle-модели | PaddlePaddle CPU, layout + OCR + таблицы |
| `dots_mocr` | dots-studio/dots.mocr | Q5_K_M + свой mmproj Q8_0 | llama.cpp, layout JSON + HTML-таблицы |
| `glm_ocr` | zai-org/GLM-OCR | Q8_0 + свой mmproj Q8_0 | llama.cpp, Text Recognition / Table Recognition |
| `chandra` | datalab-to/chandra-ocr-2 | Q5_K_M + свой mmproj Q8_0 | llama.cpp, HTML с layout |
| `lighton` | lightonai/LightOnOCR-3-1B | Q8_0 + свой mmproj Q8_0 | llama.cpp, grounding + Markdown/HTML |
| `kdl` | KDLAI/KDL-Frontier-Parser-nano | Исходные safetensors, CPU FP32 | Transformers: layout → crop → recognition → OTSL |
| `lighton_small` | lightonai/LightOnOCR-3-0.8B | Q8_0 + свой mmproj Q8_0 | llama.cpp, grounding |
| `jina` | jinaai/jina-ocr-v1 | Q8_0 + свой mmproj F16 | llama.cpp с RoPE-поправкой, Markdown |
| `surya` | datalab-to/surya-ocr-2 | Официальный F16 GGUF + mmproj | llama.cpp, HTML с layout |
| `falcon` | tiiuae/Falcon-OCR v1.5 | Исходные safetensors, CPU FP32 | Transformers с адаптацией Triton-операций под CPU |
| `nemotron` | nvidia/Nemotron-3.5-Parse | Исходные safetensors, CPU FP32 | Transformers и штатный postprocessing |

Подробности, источники весов, промпты, ограничения и параметры:
[docs/backends.md](docs/backends.md). **Качество и CPU-производительность этих моделей
в данном окружении пока не измерены**: Hugging Face недоступен для загрузки весов.
Интеграция проверяется отдельно от реального распознавания.

Paddle рендерит PDF при 250 dpi с возможным повтором на 350 dpi. Все остальные
модели используют `OCR_VLM_DPI=150` без повторного прохода на повышенном DPI.
Для llama.cpp действует `OCR_VLM_MAX_PIXELS=1500000`; dots.mocr дополнительно
ограничен `OCR_DOTS_MAX_PIXELS=1000000` (применяется меньший из двух лимитов).
Это снижает нагрузку vision encoder. PNG для просмотра сохраняется с DPI
рендера; фактический размер входа модели записывается в метрики.
Для мелкого текста увеличьте лимит dots до 1500000 и проверьте результат:
снижение разрешения не гарантирует сохранения всех деталей.
После падения llama-server проблемная страница получает ошибку без повышения
DPI, а перед следующей сервер перезапускается.

Генеративные модели не выдают сопоставимый OCR confidence: возвращаем `null` и
`needs_review=true`, не запускаем повтор только из-за отсутствия score. Ошибка
вызова/пустой ответ помечает страницу ошибкой, затем обрабатывается следующая.
Обрыв по `max_tokens` сохраняется с предупреждением. Количество ячеек и ток/с
не являются оценками точности. Для точности нужен эталон текста/таблиц.

```sh
curl -F 'file=@document.pdf' -F 'backends=paddle' -F 'backends=dots_mocr' \
  -F 'backends=glm_ocr' -F 'backends=chandra' -F 'backends=lighton' \
  -F 'backends=kdl' http://localhost:8000/api/comparisons
curl http://localhost:8000/api/comparisons/COMPARISON_ID
```

## Paddle: модели и алгоритм

Зафиксированы PaddleOCR 3.3.2, PaddleX 3.3.13 и PaddlePaddle 3.2.2 CPU.
Все Python-зависимости и их хеши зафиксированы в `uv.lock`.

| Этап | Модель |
| --- | --- |
| Ориентация страницы | PP-LCNet_x1_0_doc_ori |
| Layout | PP-DocLayout-L |
| Детекция текста | PP-OCRv5_mobile_det |
| Русский/английский OCR | eslav_PP-OCRv5_mobile_rec |
| Классификация таблицы | PP-LCNet_x1_0_table_cls |
| Структура с рамками | SLANeXt_wired |
| Структура без рамок | SLANet_plus |
| Ячейки | RT-DETR-L_wired_table_cell_det / RT-DETR-L_wireless_table_cell_det |

PDF рендерится постранично через PyMuPDF. PP-StructureV3 выполняет ориентацию,
layout, OCR и восстановление таблиц. Формулы, диаграммы, печати, unwarping и
отдельная region detection отключены; колонтитулы сохраняются. Выход адаптируется
в JSON, а HTML таблиц очищается от активного содержимого. Сохраняются объединения
ячеек, ведущие нули и пустые значения. Просмотрщик использует текстовые DOM-узлы.

Обычный рендер — 250 dpi; один повтор на 350 dpi при предупреждениях/ошибках.
Низкоуверенные фрагменты не отбрасываются. Оба исходных ответа модели сохраняются
как `page-N-DPI.raw.json`. Повтор принимается при росте средней оценки без
существенной потери текста; расхождение текста помечается для проверки. Ошибка
повтора не уничтожает успешный первый проход. Это эвристика, а не гарантия
качества: OCR score не является вероятностью правильности реквизита. Полноту
таблиц и пороги необходимо оценивать по размеченным документам.

Этот срез обрабатывает PDF через OCR, включая цифровые PDF. Прямое извлечение
текстового слоя, склейка многостраничных таблиц, бизнес-нормализация, Qwen3,
отчёт XLSX, ZIP и офисные форматы в текущий объём не входят.

## Работа без интернета

Подготовка образов и весов, экспорт/импорт тома с manifest и запуск без сборки
описаны в [гайде по изолированному развёртыванию](docs/deployment.md#8-перенести-на-машину-без-интернета).
Runtime-сервисы не имеют внешней сети и получают модели только для чтения.

Для основных GGUF закреплены revision и SHA256 из карточек артефактов.
Для Paddle/KDL/Falcon/Nemotron и dots BF16 SHA256 записывается после подготовки
и проверяется перед загрузкой; это контроль последующих изменений.
Manifest входит в результат. Для обновления моделей сначала сохраните старый
набор и подготовьте новый отдельно.

## Локальная разработка с mise

```sh
mise install
mise run install
mise run models
mise run api
# Во втором терминале:
# OCR_WORKER_BACKEND выбирает локальный worker (один backend на процесс).
OCR_WORKER_BACKEND=glm_ocr mise run worker
```

Для Nemotron локально используйте отдельное окружение `uv sync --frozen --extra nemotron`:
его extra несовместим с KDL/Falcon. Через Compose окружения разделены автоматически.
Для Jina нужен отдельный бинарник с RoPE-поправкой, включённый в её Compose-образ.
Для VLM через локальный mise/uv нужен установленный `llama-server` b10000
(в Compose он уже включён). `OCR_LLAMA_SERVER` задаёт путь к бинарнику.

Без mise эквивалентны `uv sync --frozen --extra ocr --extra kdl`,
`uv run --frozen ocr-service prepare-models --download`,
`uv run --frozen ocr-service api` и
`OCR_WORKER_BACKEND=glm_ocr uv run --frozen --extra ocr --extra kdl ocr-service worker`.

Основные настройки: `OCR_DATA_DIR`, `OCR_MODELS_DIR`, `OCR_CPU_THREADS`, `OCR_DPI`,
`OCR_RETRY_DPI`, `OCR_LAYOUT_THRESHOLD`, `OCR_REVIEW_THRESHOLD`,
`OCR_MAX_UPLOAD_MB`, `OCR_MAX_PAGES`, `OCR_MAX_PAGE_PIXELS`.
Начальные ограничения — 64 МиБ, 300 страниц и 30 млн пикселей на страницу.
PDF с паролем отклоняются до регистрации задания. Приложение сохраняет исходники,
результаты и PNG до ручной очистки; учитывайте объём диска.

## Проверки

```sh
mise run test
mise run check
# Реальный тест русского текста и таблицы, после подготовки весов:
OCR_RUN_MODEL_TESTS=1 uv run --frozen --extra ocr pytest -m ocr -v
```

Обычные тесты используют явно обозначенный тестовый движок и проверяют API,
ошибку средней страницы, продолжение следующих, восстановление очереди,
сохранение первого прохода, структуру таблиц, очистку HTML и целостность моделей.
Они не подтверждают качество настоящего OCR. Дополнительно проверяются сравнения,
HTTP-контракт мультимодального сервера, lifecycle, форматы VLM, OTSL и KDL crop routing. Реальный тест вызывает Paddle и
проверяет русское слово, номер с ведущими нулями и извлечение таблицы.

Лицензионные сведения: [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
