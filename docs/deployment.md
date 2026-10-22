# Развёртывание OCR на Linux

Гайд для запуска через Podman Compose. Python, uv, llama.cpp и зависимости OCR
устанавливаются внутри образов; отдельно ставить их на хост не нужно.
Mise можно использовать как обёртку команд проекта.

## 1. Подготовить машину

Нужны:

- Linux x86_64 и CPU с AVX2; GPU не требуется.
- Git, Podman, Compose provider и curl.
- Локальный диск для SQLite, PDF, PNG и моделей. Не размещайте том `data` на NFS/SMB.
- Интернет для сборки образов и первоначальной загрузки весов; изолированное
  развёртывание описано в разделе 8.

Например, на Debian/Ubuntu пакеты устанавливаются так:

```sh
sudo apt-get update
sudo apt-get install -y git curl podman podman-compose
```

Дальше выполняйте все команды Podman **одним обычным пользователем, без sudo**.
У rootless Podman и `sudo podman` разные образы, контейнеры и тома.

Проверьте окружение:

```sh
podman --version
podman info
podman compose version
lscpu
df -h .
```

В `lscpu` среди flags должен быть `avx2`. `podman compose` использует внешний
provider; если он не найден, установите `podman-compose`. Для явного выбора:

```sh
export PODMAN_COMPOSE_PROVIDER=podman-compose
```

Точный минимум RAM и скорость ещё не измерены на настоящих весах. RAM нужна
не только для весов, но и для vision encoder, KV-cache, render и временных
буферов. Начните с одной модели и короткого PDF. Полный набор моделей и образов
занимает десятки ГБ; дополнительно растут сборочный кеш и сохранённые страницы.

## 2. Получить проект и настроить `.env`

Текущая реализация находится в ветке `work`:

```sh
git clone --branch work https://github.com/Pheodaron/ocr.git
cd ocr
cp .env.example .env
```

Если проект уже скачан, перейдите в него. Существующий `.env` сохраняйте:
не копируйте пример поверх своих настроек.

Для первого запуска достаточно значений из примера:

```dotenv
OCR_PORT=8000
OCR_CPU_THREADS=4
OCR_DPI=250
OCR_RETRY_DPI=350
OCR_VLM_DPI=150
OCR_ALLOW_MODEL_DOWNLOAD=true
OCR_GLM_CONTEXT=16384
OCR_VLM_MAX_TOKENS=8192
OCR_VLM_MAX_PIXELS=2500000
OCR_DOTS_MAX_PIXELS=2500000
```

`OCR_CPU_THREADS` выбирайте с учётом ресурсов хоста. Для GLM не начинайте
с контекста 131072: он значительно увеличивает KV-cache и расход RAM.

`OCR_ENABLED_BACKENDS` управляет списком в API и клиенте. Оставленный полный
список **не запускает и не скачивает все модели**: это делают выбранные
Compose-профили. `OCR_WORKER_BACKEND` в `.env` предназначен для локального
worker; в Compose каждый сервис задаёт свой backend самостоятельно.

Проверьте синтаксис конфигурации:

```sh
podman compose --profile '*' config > /dev/null
```

## 3. Поднять первую модель

Пример для GLM-OCR:

```sh
podman compose --profile glm_ocr up --build -d
podman ps -a
podman compose --profile glm_ocr logs -f models-glm-ocr glm-ocr api
```

`Ctrl+C` завершает просмотр логов; контейнеры продолжают работать.

Порядок запуска:

1. Собираются API и CPU runtime выбранной модели.
2. Одноразовый `models-glm-ocr` скачивает decoder и соответствующий projector,
   проверяет файлы и сохраняет manifest в том `models`.
3. После успешной подготовки запускается worker `glm-ocr`.
4. API принимает PDF, worker забирает задания только `glm_ocr`.

Для списка контейнеров используется `podman ps -a`: некоторые версии
`podman-compose` не поддерживают флаг `-a` у своей команды `ps`.
Список включает контейнеры других проектов текущего пользователя.

`models-glm-ocr` со статусом **Exited (0)** — нормальное состояние. При ошибке
этого контейнера сначала исправьте подготовку моделей; worker зависит от неё.

Для GGUF-профилей первая сборка компилирует llama.cpp, а загрузка весов может
занять значительное время. Контейнер устанавливает зависимости из `uv.lock`
с `--no-install-project` и запускает код через `PYTHONPATH=/app/src`: отдельная
упаковка приложения и загрузка Hatchling для неё не требуются. Слой основных
Python-зависимостей создаётся до копирования исходников и переиспользуется
при изменении кода. `uv` использует системные сертификаты, проверка TLS включена. При повторных запусках используются локальные образы и сохранённые веса.
Обычный `podman compose up -d` без профиля запускает только API.

Проверка:

```sh
curl -fsS http://127.0.0.1:8000/health
curl -sS -i http://127.0.0.1:8000/ready
curl -fsS http://127.0.0.1:8000/api/config
```

В `/health` смотрите `workers.glm_ocr`. `/ready` возвращает 200, если хотя бы
один worker готов, и 503, если таких нет. Это готовность обработки очереди:
сама модель загружается в отдельном процессе при получении документа.
Поэтому окончательная проверка запуска — успешное распознавание короткого PDF.

Клиент: **http://127.0.0.1:8000**. Справка API: **http://127.0.0.1:8000/docs**.
Схема OpenAPI: **http://127.0.0.1:8000/openapi.json**.
Если изменили `OCR_PORT`, замените порт во всех адресах и командах.

## 4. Загрузить PDF и получить результат

В браузере выберите конфигурацию **GLM-OCR**, загрузите PDF и дождитесь результата.
Для начала используйте 1–3 страницы с характерным текстом и таблицей.

То же через API:

```sh
curl -fsS -F 'file=@document.pdf' -F 'backend=glm_ocr' \
  http://127.0.0.1:8000/api/documents
```

Скопируйте `id` из ответа в переменную:

```sh
JOB_ID=вставьте_id_из_ответа
curl -fsS "http://127.0.0.1:8000/api/documents/$JOB_ID"
```

Пока статус `queued` или `processing`, проверяйте этот endpoint повторно.
Терминальные статусы:

| Статус | Значение |
| --- | --- |
| `completed` | Все страницы распознаны |
| `partial` | Есть успешные и ошибочные страницы |
| `failed` | Нет успешно обработанных страниц |

Ошибка отдельной страницы не останавливает последующие. Причина сохраняется
в `pages[].error`; статус страницы и `needs_review` — разные признаки.
Для генеративного OCR `confidence=null`, `needs_review=true` ожидаемы.

После терминального статуса:

```sh
curl -fsS "http://127.0.0.1:8000/api/documents/$JOB_ID/result"
curl -fsS -o result.json \
  "http://127.0.0.1:8000/api/documents/$JOB_ID/download?format=json"
curl -fsS -o result.md \
  "http://127.0.0.1:8000/api/documents/$JOB_ID/download?format=markdown"
curl -fsS -o result.html \
  "http://127.0.0.1:8000/api/documents/$JOB_ID/download?format=html"
```

До завершения `/result` возвращает 409. Сейчас поддерживаются PDF и выдача
JSON/Markdown/HTML; ZIP, офисные документы и XLSX в этот срез не входят.

## 5. Выбрать модель или сравнить несколько

| Модель | Backend / профиль | Worker-сервис | Образ runtime |
| --- | --- | --- | --- |
| Paddle PP-StructureV3 | `paddle` | `paddle` | `paddle` |
| dots.mocr Q5_K_M | `dots_mocr` | `dots-mocr` | `vlm` |
| GLM-OCR Q8_0 | `glm_ocr` | `glm-ocr` | `vlm` |
| Chandra OCR 2 Q5_K_M | `chandra` | `chandra` | `vlm` |
| LightOnOCR-3-1B Q8_0 | `lighton` | `lighton` | `vlm` |
| KDL nano FP32 | `kdl` | `kdl` | `kdl` |
| LightOnOCR-3-0.8B Q8_0 | `lighton_small` | `lighton-small` | `vlm` |
| Jina OCR v1 Q8_0 | `jina` | `jina` | `jina` |
| Surya OCR 2 F16 | `surya` | `surya` | `vlm` |
| Falcon-OCR 1.5 FP32 | `falcon` | `falcon` | `falcon` |
| Nemotron 3.5 Parse FP32 | `nemotron` | `nemotron` | `nemotron` |

Полное имя образа: `localhost/document-ocr-<образ runtime>:0.1.0`.
Подготовка весов каждого worker называется `models-<worker-сервис>`.
Один общий образ `vlm` используется несколькими независимыми сервисами;
каждый сервис загружает только свои веса.

**Запуск нового профиля не останавливает уже работающий.** На машине
с ограниченной RAM переключайтесь явно, дождавшись завершения текущего документа:

```sh
podman compose stop glm-ocr
podman compose --profile dots_mocr up --build -d
# После завершения dots.mocr:
podman compose stop dots-mocr
podman compose --profile nemotron up --build -d
```

Для сравнения отметьте несколько моделей в клиенте и загрузите PDF один раз.
Работающий сервис выполнит свою часть, остальные задания останутся в очереди
до запуска соответствующих сервисов. После переключения результаты появятся
в том же сравнении. Автоматического переключения контейнеров нет.

Пример создания сравнения через API:

```sh
curl -fsS -F 'file=@document.pdf' \
  -F 'backends=glm_ocr' -F 'backends=dots_mocr' -F 'backends=nemotron' \
  http://127.0.0.1:8000/api/comparisons
# Подставьте id сравнения из ответа:
COMPARISON_ID=вставьте_id_сравнения
curl -fsS "http://127.0.0.1:8000/api/comparisons/$COMPARISON_ID"
```

Не включайте все профили для обычного запуска: сервисы могут загружать модели
параллельно, а их потребление RAM и CPU складывается.

## 6. Изменить настройки, остановить или обновить

Редактируйте `.env`, затем повторите `up` с нужным профилем, чтобы Compose
пересоздал контейнеры с изменившимся окружением:

```sh
podman compose --profile glm_ocr up -d
```

`restart` не применяет изменённые переменные окружения. Настройки распознавания
фиксируются при загрузке PDF: для проверки новых DPI/контекста/квантизации
загрузите документ заново. Для смены backend просто запустите его профиль.

Дополнительные параметры:

| Переменная | Назначение |
| --- | --- |
| `OCR_DOTS_QUANTIZATION` | `Q5_K_M`, `Q8_0` или `BF16`; требуется подготовка выбранных весов |
| `OCR_GLM_TASK` | `text` для страницы или `table` для табличного запроса |
| `OCR_LIGHTON_SMALL_QUANTIZATION` | prithivMLmods `Q4_K_M` по умолчанию или прежний `Q8_0`; [LightOn 0.8B на CPU](lighton-q4.md) |
| `OCR_LIGHTON_SMALL_TASK` | `transcription` (пустой промпт) или `grounding` |
| `OCR_LIGHTON_SMALL_THREADS` | 8 по умолчанию; диапазон 1–8 для LightOn 0.8B, независимо от общего `OCR_CPU_THREADS` |
| `OCR_LIGHTON_SMALL_PIN_THREADS` | true: по одному разрешённому CPU на физическое ядро; фактическая маска в engine.cpu_affinity |
| `OCR_DPI`, `OCR_RETRY_DPI` | Только PP-StructureV3: основной рендер 250 dpi и повтор 350 dpi |
| `OCR_VLM_DPI` | Все остальные модели: рендер 150 dpi, один проход без повышения DPI при ошибке |
| `OCR_VLM_MAX_PIXELS` | Максимальная площадь изображения для llama.cpp, по умолчанию 2 500 000: сохраняет A4/Letter при 150 dpi без уменьшения; preview сохраняет исходный DPI |
| `OCR_DOTS_MAX_PIXELS` | Дополнительный лимит dots.mocr, по умолчанию 2 500 000; применяется меньший из этого значения и `OCR_VLM_MAX_PIXELS` |
| `OCR_VLM_MAX_TOKENS` | Бюджет ответа; при обрыве увеличьте с учётом контекста и RAM |
| `OCR_VLM_STARTUP_SECONDS` | Ожидание загрузки llama-server, по умолчанию 300 секунд |
| `OCR_VLM_TIMEOUT_SECONDS` | Таймаут запроса генерации, по умолчанию 3600 секунд |
| `OCR_MAX_UPLOAD_MB`, `OCR_MAX_PAGES` | По умолчанию 64 МиБ и 300 страниц |

Остановить все профили и API, сохранив данные:

```sh
podman compose --profile '*' down
```

**Не добавляйте `-v` / `--volumes`**, если нужно сохранить модели и результаты.
После обычного `down` запуск того же проекта использует прежние именованные тома.
При прерывании документа завершённые страницы сохраняются, незавершённые
восстанавливаются в пределах лимита попыток. При SIGTERM worker даёт текущей
странице до 10 секунд на завершение, затем останавливает группу процессов.
Compose даёт контейнеру 120 секунд до принудительного завершения.

Для остановки конкретного задания используйте «Остановить» в карточке,
истории или строке сравнения. Перезапуск API не отменяет очередь: задания
хранятся в SQLite. Кнопка прерывает текущую страницу, включая llama-server,
а готовые страницы сохраняет. Статус меняется `cancelling` → `cancelled`;
после остановки worker берёт следующий документ. Запрос остановки переживает
перезапуск и не восстанавливается как незавершённое распознавание.

«Удалить» работает с любым статусом, включая ошибки и зависшую обработку.
Задание сразу скрывается из списков, затем удаляются исходный PDF, PNG и raw
результаты. Работающее задание сначала останавливается. Удаление одного
прогона сравнения не удаляет PDF у остальных. Если worker отсутствует,
API самостоятельно завершает отмену/удаление после проверки блокировки.

Те же действия через API:

```sh
curl -X POST http://127.0.0.1:8000/api/documents/ID/cancel
curl -X DELETE http://127.0.0.1:8000/api/documents/ID
```

Остановка возвращает 202 и статус задания. Удаление возвращает 200 `deleted`
либо 202 `deleting`, пока worker останавливает процесс и очищает файлы.
После удаления GET задания возвращает 404. Готовые страницы `cancelled`
задания доступны через обычные endpoints результата и скачивания.

Обновление исходников в установленном Git checkout:

```sh
podman compose --profile '*' down
git pull --ff-only origin work
podman compose --profile glm_ocr up --build -d
```

Перед обновлением сохраните резервную копию из раздела 7. Теги образов сейчас
постоянные, поэтому для обновления приложения нужен `--build`. `.env` сохраняйте;
новые параметры сверяйте с `.env.example`. Не подменяйте веса у выполняющихся
заданий; обновление модели проводите отдельно от сравнения старых результатов.

## 7. Где данные и как сделать резервную копию

При стандартном имени проекта `document-ocr` используются тома:

| Том | Содержимое |
| --- | --- |
| `document-ocr_data` | SQLite, исходные PDF, результаты, PNG, raw-ответы, логи и runtime-кеш |
| `document-ocr_models` | Веса и manifest-файлы с контрольными суммами |

Имена относятся к текущему пользователю Podman. Если задавали `-p` или
`COMPOSE_PROJECT_NAME`, проверьте реальные имена через `podman volume ls`.
Данные не сохраняются автоматически в каталог `./data` на хосте.

Для согласованной копии SQLite сначала остановите API и все workers:

```sh
podman compose --profile '*' stop
mkdir -p backup
podman volume export document-ocr_data --output backup/data.tar
podman volume export document-ocr_models --output backup/models.tar
cp .env backup/pipeline.env
git rev-parse HEAD > backup/revision.txt
podman compose --profile glm_ocr up -d
```

Восстановление на **чистой установке**, где эти тома ещё не содержат данных:

```sh
podman volume create document-ocr_data
podman volume create document-ocr_models
podman volume import document-ocr_data backup/data.tar
podman volume import document-ocr_models backup/models.tar
cp backup/pipeline.env .env
```

Используйте совместимую версию приложения и образы. Импорт архива в уже
используемый том не заменяет аккуратное восстановление: сначала остановите
приложение и сохраните его текущее содержимое отдельно.
Автоматической очистки PDF/PNG и результатов сейчас нет.

## 8. Перенести на машину без интернета

Ниже пример для GLM и Nemotron: образы обоих вариантов переносятся заранее,
но на целевой машине модели запускаются по одной. Podman и Compose provider
должны быть установлены на целевой машине заранее. Архитектура обеих машин — x86_64.

### На машине с интернетом

Из каталога проекта:

```sh
podman compose --profile glm_ocr --profile nemotron build
OCR_ALLOW_MODEL_DOWNLOAD=true podman compose --profile glm_ocr run --rm models-glm-ocr
OCR_ALLOW_MODEL_DOWNLOAD=true podman compose --profile nemotron run --rm models-nemotron
mkdir -p offline/images
for image in api vlm nemotron; do
  podman save --format oci-archive \
    --output "offline/images/ocr-$image.tar" \
    "localhost/document-ocr-$image:0.1.0"
done
podman volume export document-ocr_models --output offline/models.tar
git archive --format=tar --output=offline/project.tar HEAD
cp .env offline/pipeline.env
git rev-parse HEAD > offline/revision.txt
```

Каждый образ записывается в отдельный OCI-архив. Веса экспортируются отдельно
вместе с manifest и владельцами файлов; не нужно вручную копировать отдельные GGUF.
Для других профилей подготовьте их `models-*` и добавьте runtime-образ из таблицы.
`api` нужен всегда. Для dots, Chandra, обеих LightOn и Surya используется
тот же `vlm`; для Jina нужен собственный образ `jina`.

Перенесите весь каталог `offline/` доступным вам способом.

### На чистой изолированной машине

Задайте путь к перенесённому каталогу и создайте каталог приложения:

```sh
OCR_OFFLINE_BUNDLE=/путь/к/offline
mkdir -p ~/document-ocr
cd ~/document-ocr
tar -xf "$OCR_OFFLINE_BUNDLE/project.tar"
cp "$OCR_OFFLINE_BUNDLE/pipeline.env" .env
for archive in "$OCR_OFFLINE_BUNDLE"/images/*.tar; do
  podman load --input "$archive"
done
podman volume create document-ocr_models
podman volume import document-ocr_models "$OCR_OFFLINE_BUNDLE/models.tar"
```

В `.env` установите `OCR_ALLOW_MODEL_DOWNLOAD=false`, затем:

```sh
OCR_ALLOW_MODEL_DOWNLOAD=false podman compose --profile glm_ocr up -d --no-build
podman compose --profile glm_ocr logs -f models-glm-ocr glm-ocr
```

Подготовка теперь только проверяет локальные файлы. `--no-build` исключает
сборку; все нужные образы уже должны быть загружены. Workers работают без внешней
сети. Для переключения на Nemotron:

```sh
podman compose stop glm-ocr
OCR_ALLOW_MODEL_DOWNLOAD=false podman compose --profile nemotron up -d --no-build
```

Для этого варианта `.git` не переносится: обновления доставляются новым архивом
проекта, образами и, при необходимости, отдельно подготовленными весами.

## 9. Диагностика

| Симптом | Что проверить |
| --- | --- |
| API доступен, задание постоянно `queued` | Запущен ли сервис именно выбранного backend; `/health` → `workers` и `podman ps -a` |
| `models-*` завершился с ненулевым кодом | Его логи, доступ к источнику весов, свободный диск; в offline-режиме — импортирован ли правильный том |
| `service_completed_successfully` не поддерживается | Обновите Compose provider; worker должен ждать успешной подготовки весов |
| `Permission denied` на томе | Не смешаны ли `podman` и `sudo podman`, корректно ли импортированы владельцы файлов; не делайте `chmod 777` |
| `Illegal instruction` | Архитектура x86_64 и наличие AVX2 у CPU/виртуальной машины |
| Процесс убит / exit 137 | Проверьте RAM и cgroup memory.events; exit 137 контейнера не следует путать с кодом -9 дочернего llama-server. SIGKILL сам по себе ещё не подтверждает OOM |
| `llama-server завершился` | В pages[].error есть код/сигнал и путь к журналу; проблемная страница не повторяется на повышенном DPI, перед следующей сервер перезапускается |
| `download model from bos` при загрузке Paddle | Обновите код и пересоберите `paddle`: исправлен безусловный запуск выключенного PP-Chart2Table в PaddleX 3.3.13; после обновления загрузите PDF заново |
| `Failed to fetch pypi.org` / `tls handshake eof` при сборке | Сбой TLS/сети при скачивании пакета: проверьте доступ к PyPI, прокси и доверенные CA; пересборка приложения больше не скачивает Hatchling, но зависимости всё ещё требуют сеть |
| HTTP 404 при загрузке dots Q5_K_M / BF16 | Обновите код и пересоберите профиль `dots_mocr`, включая API-образ загрузчика и VLM-образ worker: исправлен revision, не содержавший эти файлы |
| Ошибка SHA256 | Файл повреждён или не соответствует pinned revision; подготовьте правильные веса в отдельном наборе, не редактируйте manifest вручную |
| `llama-server` не успел стартовать | Лог worker и `/data/logs/llama-<backend>.log`; RAM и `OCR_VLM_STARTUP_SECONDS` |
| Обрыв ответа / `finish_reason=length` | Бюджет `OCR_VLM_MAX_TOKENS`, контекст и сложность страницы; результат помечен предупреждением |
| HTTP 422 при загрузке | Поле `file`, выбранный backend, корректность PDF и лимиты; PDF с паролем не поддерживается |
| HTTP 413 | Размер превышает `OCR_MAX_UPLOAD_MB` |
| HTTP 409 на `/result` | Обработка ещё не завершена; проверяйте статус документа |
| Порт занят | Измените `OCR_PORT` в `.env` и пересоздайте API через `up -d` |
| Закончилось место при сборке/работе | `df -h`, `podman system df`, размер томов и сборочного кеша; удаляйте только ненужные артефакты, сохранив данные |

Логи конкретной модели:

```sh
podman ps -a
podman compose --profile nemotron logs --tail=100 models-nemotron nemotron api
```

Если dots.mocr работает слишком долго или убит OOM-killer, начните новый
прогон с ограничением входа и меньшим контекстом. В `.env`:

```dotenv
OCR_VLM_MAX_PIXELS=2500000
OCR_VLM_DPI=150
OCR_DOTS_MAX_PIXELS=2500000
OCR_DOTS_CONTEXT=16384
```

Обновите оба образа (API сохраняет настройки задания, worker их использует),
пересоздайте сервисы и загрузите PDF заново:

```sh
podman compose --profile dots_mocr up --build --force-recreate -d
```

A4 при 150 dpi даёт примерно 1241×1754 px (2,18 млн пикселей). Лимиты выше
сохраняют такую страницу целиком, без дополнительного уменьшения. Это
исходный режим для текста 10–12 pt; точность зависит от качества скана.
Не оставляйте старые значения 1000000/1500000 в `.env`: они уменьшат A4
ниже выбранных 150 dpi. Для остальных llama.cpp моделей действует общий
лимит; KDL/Falcon/Nemotron используют свои processors.
Для мелкого текста можно поднять DPI до 200, но оба лимита для dots тогда
должны быть не ниже 4000000, чтобы сохранить A4 без уменьшения.
Это увеличивает нагрузку и RAM. Генерация длинного JSON/HTML на CPU тоже
может занимать минуты независимо от разрешения изображения.

Не рассчитывайте RAM только по размеру GGUF: vision encoder,
attention и KV-cache требуют дополнительной памяти. Ранее завершённые ошибкой
задания автоматически не переобрабатываются.

Для dots.mocr журнал сервера и лимиты памяти можно получить так:

```sh
podman compose --profile dots_mocr exec dots-mocr tail -n 120 /data/logs/llama-dots_mocr.log
podman compose --profile dots_mocr exec dots-mocr sh -c 'cat /sys/fs/cgroup/memory.max; cat /sys/fs/cgroup/memory.events'
free -h
```

Чтобы разделить обработку изображения и входных токенов декодера, включите
расширенный журнал готовым override (пересборка образа не нужна):

```sh
podman compose -f compose.yaml -f compose.debug.yaml --profile dots_mocr up -d --force-recreate dots-mocr
podman compose --profile dots_mocr exec dots-mocr tail -n 200 -f /data/logs/llama-dots_mocr.log
```

`compose.debug.yaml` передаёт `LLAMA_ARG_LOG_VERBOSITY=4` и
`LLAMA_ARG_LOG_TIMESTAMPS=true` непосредственно в контейнер worker;
llama-server наследует эти переменные. Одна запись переменных в `.env` без
их передачи через Compose не включает диагностику. Уровень 4 показывает
низкоуровневые INFO-сообщения библиотек llama.cpp, скрытые на уровне 3.

| Сообщение в журнале | Этап |
| --- | --- |
| `flash attention is enabled` / `disabled` | Фактический режим визуального энкодера |
| `encoding mtmd batch from idx` | Начало вычисления визуальных embeddings |
| `decoding image batch 1/...` | Embeddings готовы; начинается подача визуальных токенов в декодер |
| `image decoded (batch .../...) in ... ms` | Время одного пакета визуальных токенов |
| `n_decoded`, `tg` | Генерация выходного текста и её скорость |

Пауза между `encoding mtmd batch` и первым `decoding image batch` относится
к визуальному энкодеру. Медленные пакеты `image decoded` указывают на
обработку входного контекста декодером. Время подготовки входа и время
генерации также доступны в `pages[].metrics.timings` после завершения страницы.
Для чистого замера остановите прежнее задание кнопкой и загрузите одну страницу.
Уровень 5 пишет значительно больше данных, включая содержимое запросов;
для первичной диагностики достаточно 4.

Вернуть обычный уровень журнала:

```sh
podman compose --profile dots_mocr up -d --force-recreate dots-mocr
```

У `exec` тоже указывайте профиль: некоторые версии podman-compose иначе
не видят сервис. На cgroup v2 `memory.max=max` означает отсутствие собственного
жёсткого лимита на этом уровне; лимит предков/хоста всё ещё возможен.
Рост `oom_kill` в `memory.events` подтверждает OOM-убийства в этой cgroup,
но счётчик накопительный за время её существования. Если этих файлов нет,
проверьте cgroup/системный журнал хоста. Свободная RAM после сбоя не показывает
пик памяти непосредственно перед ним. Предупреждения об отсутствии GPU
ожидаемы для CPU-сборки.

После исправления ошибки инициализации перезапустите нужный сервис и загрузите
PDF повторно: уже завершённое `failed`-задание автоматически не переотправляется.
При обычном перезапуске worker восстанавливает свои незавершённые задания.

Порт API опубликован только на `127.0.0.1`; для удалённого просмотра можно
использовать SSH-туннель:

```sh
ssh -L 8000:127.0.0.1:8000 user@ocr-host
```

Откройте http://127.0.0.1:8000 на локальном компьютере. Встроенной аутентификации
в приложении нет; прямую публикацию API во внешнюю сеть этот гайд не настраивает.

## 10. Mise и границы проверки

Для Compose установка Python через mise не обязательна. Если mise уже настроен:

```sh
mise trust
mise install
OCR_WORKER_BACKEND=nemotron mise run up
# Для остановки всех профилей используйте команду из раздела 6.
```

`mise run install` устанавливает локальные Python-зависимости для разработки;
для контейнерного запуска этот шаг не нужен. Локальная разработка без контейнеров
описана в [README](../README.md#локальная-разработка-с-mise).

Сборки и контракты проверены, но настоящее распознавание всех моделей и их
CPU-производительность ещё не подтверждены: загрузка весов заблокирована
в окружении разработки. В частности, Falcon использует отдельную CPU-адаптацию,
Nemotron — новый custom runtime. Не принимайте HTTP 202 или успешную сборку
за подтверждение OCR-качества. Детали: [проверки](validation.md),
[алгоритмы и ограничения моделей](backends.md), [лицензии](../THIRD_PARTY_NOTICES.md).
