# OCR-конфигурации

## Общий контракт

`POST /api/documents`: multipart `file` + `backend`.
`POST /api/comparisons`: один `file`, повторяющееся поле `backends`.
Допустимые ID публикует `/api/config`; набор ограничивает `OCR_ENABLED_BACKENDS`.
Нельзя передавать произвольный URL модели или исполняемый код через API.

Исходник сравнения хранится через hard links, поэтому байты PDF одинаковы.
Все задания группы публикуются одной транзакцией SQLite после сохранения файлов.
Каждый backend — отдельный сервис и профиль в Compose. API запускается
независимо. Подготовка `models-*` скачивает только веса выбранного профиля.
Runtime-контейнеры не требуют сетевого соединения друг с другом или с API:
задания/результаты хранятся в общем локальном томе SQLite.
Каждый worker фильтрует очередь по своему backend; блокировка и восстановление
после сбоя также ограничены этим backend. `/health` содержит отдельные `workers`.

Внутри сервиса inference-процесс запускается на документ и завершается после него.
Для GGUF он запускает собственный `llama-server` на loopback в том же контейнере
`network_mode: none`. API не импортирует Paddle/PyTorch. Для ограниченной RAM
профили включают по одному: остановить текущий сервис, запустить следующий.
Задания других моделей продолжают ждать; результаты сравнения сохраняются.
Если оператор включит несколько профилей, сервисы смогут работать параллельно;
общего автоматического переключателя моделей или лимитера RAM нет.

Снимок render/model/context/CPU-параметров сохраняется при загрузке PDF. Поэтому
изменение `.env` не меняет уже стоящие в очереди задания. Исторический результат
сохраняет свою квантизацию и контекст, независимо от текущего конфига сервера.
Задания из предыдущей версии без отдельных DPI/лимита dots сохраняют прежний
DPI и общий лимит изображения. Новые значения требуют новой загрузки PDF.
Модели одного backend при обновлении следует хранить отдельно от старого набора.

Все ответы адаптируются в `PageResult`: text, markdown, blocks, tables/cells,
warnings, metrics. Содержимое HTML-таблиц проходит общий sanitizer.
Строковые значения ячеек, пустые значения и объединения не преобразуются в числа.
Raw JSON сохраняется в `data/documents/ID/page-N-DPI.raw.json`.

Рендер зависит от backend: Paddle использует `OCR_DPI=250` и при необходимости
`OCR_RETRY_DPI=350`; все остальные модели — `OCR_VLM_DPI=150`, один проход
без повышения DPI даже при ошибке/пустом ответе. Это DPI рендера PDF, а не
восстановление детализации исходного скана. KDL, Falcon и Nemotron далее
применяют свои штатные image processors.

Paddle выдаёт OCR score. У генеративных моделей `confidence=null`,
`needs_review=true`; это не ошибка страницы и не основание повторно вызывать
модель на 350 dpi.
Ошибка страницы не прерывает следующие страницы и другие конфигурации.
Если llama-server завершился или оборвал транспортный запрос, текущая страница
получает ошибку без повторного запроса на повышенном DPI. Перед следующей
страницей процесс перезапускается. Ошибка содержит код завершения/сигнал,
путь к журналу и ограниченный фрагмент сообщений текущего процесса.
Если сам перезапуск не удался, эта причина сохраняется для оставшихся страниц
без повторной загрузки модели на каждой странице. Время успешных перезапусков
входит в processing_seconds; pages[].metrics.server_restarts показывает их число.
Ошибка инициализации модели завершает только её задание с явным сообщением.
Повторные native-crash при загрузке также ограничены, чтобы очередь не зациклилась.

## Веса и runtime

| Backend | Исходная модель | Поставка CPU |
| --- | --- | --- |
| `paddle` | [PP-StructureV3](https://www.paddleocr.ai/main/en/version3.x/pipeline_usage/PP-StructureV3.html) | Девять inference-моделей Paddle из официального BOS registry |
| `dots_mocr` | [dots-studio/dots.mocr](https://huggingface.co/dots-studio/dots.mocr) | [GGUF lodrick-the-lafted](https://huggingface.co/lodrick-the-lafted/dots.mocr-gguf), Q5_K_M, Q8_0 или BF16; mmproj Q8_0 |
| `glm_ocr` | [zai-org/GLM-OCR](https://huggingface.co/zai-org/GLM-OCR) | [GGUF ggml-org](https://huggingface.co/ggml-org/GLM-OCR-GGUF), decoder и mmproj Q8_0 |
| `chandra` | [datalab-to/chandra-ocr-2](https://huggingface.co/datalab-to/chandra-ocr-2) | [GGUF prithivMLmods](https://huggingface.co/prithivMLmods/chandra-ocr-2-GGUF), Q5_K_M + mmproj Q8_0 |
| `lighton` | [lightonai/LightOnOCR-3-1B](https://huggingface.co/lightonai/LightOnOCR-3-1B) | [GGUF prithivMLmods](https://huggingface.co/prithivMLmods/LightOnOCR-3-1B-GGUF), decoder и mmproj Q8_0 |
| `kdl` | [KDLAI/KDL-Frontier-Parser-nano](https://huggingface.co/KDLAI/KDL-Frontier-Parser-nano) | Исходные safetensors, Transformers 5.5.0 + PyTorch 2.7.1+cpu, FP32 |

GGUF decoder без соответствующего vision encoder/projector не распознаёт картинки.
Файлы разных моделей не смешиваются. В `vlm_models.py` зафиксированы HF revision,
имена и опубликованные SHA256 основных артефактов. BF16 dots и KDL фиксируются
локальным manifest после подготовки; все файлы проверяются перед загрузкой.
Смена decoder dots на BF16 не меняет точность его Q8_0 vision projector.

Compose собирает CPU `llama-server` из
[llama.cpp b10000](https://github.com/ggml-org/llama.cpp/tree/47a39665e7081dc482feec169961acc09750a5c4),
commit `47a39665e7081dc482feec169961acc09750a5c4`.
CUDA отключена при сборке; runtime передаёт `--n-gpu-layers 0`,
`--no-mmproj-offload`, `--parallel 1`, `--offline`, `--fit off`.
HTTP-запрос передаёт локальный PNG как base64 data URL и затем промпт;
`temperature=0`, `seed=42`, `cache_prompt=false`. У каждой страницы отдельный
chat-контекст; страницы не добавляются в одну длинную беседу.

До PNG/base64 вход llama.cpp ограничивается `OCR_VLM_MAX_PIXELS` (по умолчанию
2 500 000 пикселей) с сохранением пропорций. Для dots.mocr применяется меньший
из общего лимита и `OCR_DOTS_MAX_PIXELS` (по умолчанию 2 500 000).
A4/Letter при 150 dpi помещаются в этот бюджет без дополнительного уменьшения:
A4 даёт примерно 1241×1754 px. Это базовый режим для среднего текста 10–12 pt;
он сохраняет разрешение страницы, но не гарантирует точность OCR на плохом скане.
Это отдельный лимит от DPI рендера:
исходный preview остаётся в полном разрешении, bbox в результатах относятся к нему.
Метрики `image_preprocessing` содержат размеры render и фактического входа;
уменьшение видно в предупреждении страницы и в raw. Низкий лимит может ухудшать
мелкий текст; увеличение требует повторного измерения RAM. Лимит не относится
к Paddle и Transformers-backend, у которых собственная предобработка.

Пользовательский запуск dots.mocr с исходной полноразмерной страницей привёл
к OOM-убийству (`memory.events: oom_kill=1`) на хосте с 31 GiB RAM. Модель при
этом успела загрузиться, а процесс погиб при обработке первого изображения.
Ограничение изображения снижает нагрузку vision encoder; оно не гарантирует
одинаковый расход памяти для всех моделей и документов.

### Совместимость PaddleX 3.3.13

В этой версии конструктор создаёт `PP-Chart2Table` даже при выключенных
диаграммах. `paddle_runtime.py` пропускает только выключенный chart predictor
и проверяет локальные пути остальных моделей до вызова factory. Контекст
защиты охватывает инициализацию и predict, чтобы охватить ленивый OCR таблиц.
Глобальная factory восстанавливается после вызова; каждый документ имеет
свой отдельный процесс. Скачивать дополнительную модель или открывать сеть
OCR-контейнеру не требуется.

## Алгоритмы новых обработчиков

### dots.mocr

PDF → BGR render → PNG → собственный vision encoder + decoder → layout JSON.
Используется официальный `prompt_layout_all_en` из
[dots.ocr prompts.py](https://github.com/rednote-hilab/dots.ocr/blob/master/dots_ocr/utils/prompts.py),
сохранённый в `dots_prompt.txt`. Модель возвращает категории блоков, их текст
и HTML для таблиц. Адаптер сохраняет порядок ответа; принимает JSON-массив
блоков или объект с `layout`/`blocks`/`elements`.

Если JSON обрезан/невалиден, исходный текст сохраняется с предупреждением, без
выдумывания структуры таблиц. Сгенерированные bbox остаются в raw: их система
координат зависит от внутреннего resize llama.cpp, поэтому `block.bbox=null`.
Для Q5_K_M и BF16 закреплён commit
[`8bc6c189780084cf59b10d55fadcbe7d1292696a`](https://huggingface.co/lodrick-the-lafted/dots.mocr-gguf/tree/8bc6c189780084cf59b10d55fadcbe7d1292696a),
где опубликованы эти файлы. Исходный `fc5b04f...` содержит только Q8 decoder
и Q8 projector; он сохранён для Q8 ради совместимости существующих manifest.
SHA256 Q5 decoder сверена с [LFS pointer в коммите загрузки](https://huggingface.co/lodrick-the-lafted/dots.mocr-gguf/commit/8bc6c189780084cf59b10d55fadcbe7d1292696a).
Ориентация отдельно не классифицируется. Для повёрнутых сканов качество проверяется
на том же тестовом наборе, что и у других VLM.

### GLM-OCR

PDF → render → multimodal decoder → ответ на один из штатных промптов:
`Text Recognition:` (`OCR_GLM_TASK=text`, по умолчанию) либо
`Table Recognition:` (`OCR_GLM_TASK=table`). Вторая конфигурация предназначена для
страниц с таблицами; её полноту извлечения обычного текста необходимо проверять.
HTML-таблицы разбираются в общий массив cells. bbox отсутствуют.

Это прямой вызов GLM-OCR на странице. Он не воспроизводит официальный SDK,
который может отдельно использовать layout detector и распознавание crop-областей.
Поэтому результаты рейтинга SDK нельзя автоматически переносить на этот маршрут.

В [исходном config.json](https://huggingface.co/zai-org/GLM-OCR/blob/main/config.json)
`max_position_embeddings=131072`. По умолчанию выделяется 16384, чтобы не занимать
RAM большим KV-cache. Для полного контекста:

```dotenv
OCR_GLM_CONTEXT=131072
```

Это размер входа + выхода + image tokens, а не число разрешённых выходных токенов.
Лимит выхода отдельно задаёт `OCR_VLM_MAX_TOKENS`. Ключи KV по умолчанию Q8_0,
значения F16, flash attention отключён. При 131072 у GLM только такой KV-cache
занимает приблизительно 6.1 GiB, без весов, vision encoder и вычислительных буферов.
Утверждения «10 ГБ» и «>5 ток/с» не считаются проверенным профилем приложения.

### Chandra OCR 2

PDF → render → multimodal decoder с официальным `ocr_layout`-промптом → HTML.
Промпт из [chandra/prompts.py](https://github.com/datalab-to/chandra/blob/master/chandra/prompts.py)
сохранён в `chandra_prompt.txt`. Layout-блоки используют `data-label` и
`data-bbox` в 0–1000; корректные координаты переводятся в пиксели preview.
HTML-таблицы очищаются и разбираются отдельно. Остальной HTML выводится как текст,
активное содержимое не исполняется. Исходный HTML остаётся в raw-ответе.

### LightOnOCR-3-1B

Используется именно версия **3-1B**, а не 2-1B, 3-0.8B или 3-4B.
Архитектура 1B совместима с предыдущей LightOnOCR-2, веса новые.
Штатный промпт `grounding` возвращает Markdown с маркерами
`![type](x1,y1,x2,y2)` и таблицами в HTML. Метки в диапазоне 0–1000 переводятся
в пиксельные bbox. Содержимое после маркера связывается с соответствующим блоком.
Неисправные bbox остаются `null`; текст сохраняется. Предобработка vision encoder
выполняется llama.cpp; используемый DPI сохраняется в результате.

### KDL-Frontier-Parser-nano

Здесь используется исходный CPU checkpoint, без неподтверждённой GGUF-конверсии.
1. Страница рендерится и приводится к RGB. Для layout создаётся изображение
   1036×1036, как в [публичном ParseBench provider](https://github.com/run-llama/ParseBench/blob/main/src/parse_bench/inference/providers/parse/kdl_frontier_nano.py).
2. Запрос `\nLayout Detection:\n`, greedy decoding. Специальные токены сохраняются.
3. Разбираются `<|box_start|>x1 y1 x2 y2<|box_end|><|ref_start|>category<|ref_end|>`
   и необязательный `<|rotate_*|>`. Координаты 0–1000 переводятся в размеры
   исходного render, ограничиваются проверками геометрии и числа регионов.
4. Каждый bbox вырезается из исходной страницы. При необходимости crop поворачивается.
5. Последовательно вызываются `Text Recognition`, `Table Recognition`,
   `Formula Recognition` или `Image Analysis` со штатными переводами строк.
6. OTSL таблиц преобразуется в HTML. `fcel`/`ecel` создают заполненную/пустую
   ячейку, `lcel` продолжает слева, `ucel` сверху, `xcel` в обе стороны,
   `nl` завершает строку. Непрямоугольные/неполные объединения отклоняются.
   Вложенные OTSL не интерпретируются; их raw сохраняется для проверки.
7. Ошибка crop сохраняется как предупреждение; следующие регионы обрабатываются.
   Если не распознан ни один регион, страница считается ошибочной.

`enable_thinking=false`, `skip_special_tokens=false`, CPU FP32/SDPA,
генерация с KV-cache. Контекст 8192; бюджет выхода уменьшается на длину входа.
Число областей ограничивает `OCR_KDL_MAX_REGIONS` (150), вход vision encoder —
`OCR_KDL_MAX_PIXELS` (1003520). Этот маршрут повторяет базовые этапы, но не все
эвристики объединения регионов и очистки публичного benchmark provider.
Результат рейтинга не является измеренным результатом этого приложения.

## Измерения

- `document.model_load_seconds`: проверка целостности весов и запуск модели.
- `document.processing_seconds`: render, inference, адаптация и запись страниц;
  без ожидания очереди и загрузки модели. При возобновлении складывается.
- `pages[].seconds`: время страницы с повторами.
- `pages[].passes`: каждый проход, DPI, ошибки и метрики.
- `pages[].metrics.usage`: токены входа/выхода, когда runtime их сообщает.
- `pages[].metrics.timings`: статистика llama.cpp. UI считает скорость декодирования
  как сумму `predicted_n` / сумму `predicted_ms`, а не tokens / общее время страницы.
- KDL сохраняет число вызовов, суммарные токены и время generate; оно включает
  prefill, поэтому UI не выдаёт его за скорость декодирования.

Сравнивайте исходный PDF с результатами, особенно числа, ведущие нули,
объединения и переносы строк. Токены у разных моделей различаются. Сумма ячеек
не показывает правильность таблицы. Для воспроизводимого сравнения фиксируйте
модели, CPU threads, DPI, task, context и prompt из результата.

## Лицензии

dots/GLM — MIT; LightOnOCR-3 — Apache-2.0; KDL — AGPL-3.0 по карточке модели.
Веса Chandra имеют modified OpenRAIL-M: автор ограничивает коммерческое
использование, в том числе условиями funding/revenue и конкуренции с API.
Исходный код Chandra — Apache-2.0; это не лицензия её весов.
Подробности и первоисточники — [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).

## Дополнительные модели

| Backend / профиль | Модель | Веса / образ |
| --- | --- | --- |
| `lighton_small` | [lightonai/LightOnOCR-3-0.8B](https://huggingface.co/lightonai/LightOnOCR-3-0.8B) | [GGUF prithivMLmods](https://huggingface.co/prithivMLmods/LightOnOCR-3-0.8B-GGUF), decoder и свой mmproj Q8_0; образ `vlm` |
| `jina` | [jinaai/jina-ocr-v1](https://huggingface.co/jinaai/jina-ocr-v1) | [GGUF ahmadw](https://huggingface.co/ahmadw/jina-ocr-v1-GGUF), Q8_0 + mmproj F16; отдельный образ `jina` |
| `surya` | [datalab-to/surya-ocr-2](https://huggingface.co/datalab-to/surya-ocr-2) | [Официальный GGUF](https://huggingface.co/datalab-to/surya-ocr-2-gguf), decoder и mmproj F16; образ `vlm` |
| `falcon` | [tiiuae/Falcon-OCR](https://huggingface.co/tiiuae/Falcon-OCR), v1.5 | Исходные safetensors, FP32; отдельный образ `falcon` |
| `nemotron` | [nvidia/Nemotron-3.5-Parse](https://huggingface.co/nvidia/Nemotron-3.5-Parse) | Исходные safetensors, FP32; отдельный образ `nemotron` |

### LightOnOCR-3-0.8B

Самостоятельный checkpoint на базе Qwen3.5. Он не использует decoder или projector
версии 1B. Маршрут: render → llama.cpp → `grounding` → Markdown/HTML и нормализованные
bbox. Адаптер общий с 1B, веса и сервисы разные.

### Jina OCR v1

DeepSeek OCR MoE: число активных параметров не равно размеру хранимых весов.
Используются исправленные Q8_0-веса от 12 октября, pinned revision
`333253b2ff8f3d9357d155327bee5dc98e007829`.
В отдельной сборке llama.cpp удалена устаревшая проверка `rope_theta == 10000`
в `deepseek2.cpp`: модель задаёт 1000000, а операция RoPE уже принимает значение
из конфига. Изменение явно записывается в metadata как `jina-rope-theta`.
Остальные модели используют стандартный бинарник.

Промпт просит транскрипцию страницы в Markdown с порядком чтения. HTML и pipe
таблицы разбираются в cells; ведущие нули и пустые значения сохраняются.
FastMTP из исходного runtime в GGUF-маршруте не используется. Лицензия исходной
модели **CC-BY-NC-4.0**, поэтому этот вариант не подходит для произвольного
коммерческого внедрения без отдельного разрешения правообладателя.

### Surya OCR 2

Вместо `onnx-community/Surya-Ocr-2-Onnx`, где нет опубликованных ONNX-весов,
подключена выбранная исходная модель Datalab. Используется её официальный
F16 GGUF-runtime на CPU, не ONNX и не сторонняя модель.
Промпт запрашивает HTML: div с `data-label`, `data-bbox` (0–1000) и содержимым.
Адаптер переводит bbox в пиксели и разбирает HTML-таблицы, как для Chandra.
Это один вызов на страницу; дополнительные crop-проходы полного Surya SDK
не воспроизводятся. Код Apache-2.0; веса имеют modified OpenRAIL с условиями
коммерческого использования, включая порог funding/revenue $5M; см. карточку.

### Falcon-OCR 1.5

Оригинальная реализация использует Triton, FlexAttention и CUDA-оптимизацию.
Для CPU подготовлена отдельная адаптация локальной копии pinned Python-кода:

1. Triton gate заменяется эквивалентным `ReLU(gate)^2 * up` для interleaved весов.
2. Attention вычисляется обычным PyTorch по блокам из 64 query-позиций.
   Сохраняются causal/document/left-pad/image-prefix masks и sink scaling.
3. Неиспользуемый GPU layout loader исключается. Второй checkpoint layout
   не загружается. Вызывается штатный whole-page `generate`, `category=plain`,
   greedy, seed 42, `compile=false`, максимальная сторона 1024.
4. Исходные веса остаются неизменными и проверяются manifest; модифицированный
   код находится в `/data/cache`, отдельно от read-only snapshot.

CPU-математика проверена тестами относительно PyTorch SDPA. Полный запуск
с настоящими весами **не проверен** из-за блокировки загрузки; CPU-порт нельзя
считать подтверждённым результатом рейтинга. Уменьшение страницы до 1024 может
ухудшать мелкий текст. Upstream `generate` не сообщает причину завершения,
поэтому raw содержит `finish_reason=unknown` и предупреждение. Лицензия Apache-2.0.

### Nemotron 3.5 Parse

Используется именно релиз **3.5**, checkpoint
`nvidia/Nemotron-3.5-Parse`, revision
`41897c0ae2fe5d9b461b207bb13587ed54d2740c`.
Это C-RADIOv4-SO400M vision encoder и десятислойный mBART decoder с GQA.

Алгоритм: PDF render → RGB → штатный AutoProcessor с динамической прямоугольной
предобработкой NaViT → AutoModel на CPU FP32/eager → greedy generation →
штатный `postprocess_document` → Markdown с HTML-таблицами и structured blocks.
Используется управляющий промпт
`</s><s><predict_bbox><predict_classes><output_markdown><predict_text_in_pic>`.
Лимит выхода — минимум из `OCR_VLM_MAX_TOKENS` и 8744; применяется generation
max_time. Исходная сериализация и блоки с bbox сохраняются в raw JSON.
HTML-таблицы проходят общий sanitizer и преобразуются в cells с объединениями.
Это один проход на страницу, без triple-pass/region-union из ParseBench.

Изолированный образ содержит Transformers 5.14.1, PyTorch 2.11.0+cpu,
torchvision 0.26.0+cpu, timm 1.0.22 и BeautifulSoup 4.13.3. Он отделён от
KDL/Falcon с другой версией Transformers. CUDA, vLLM и MTP не используются.
NVIDIA публикует GPU-примеры; CPU-интеграция и скорость на реальном checkpoint
в этом окружении ещё не подтверждены. Лицензия OpenMDW-1.1, tokenizer MIT;
атрибуции скачиваются вместе с pinned snapshot.
