# Развёртывание OCR на Linux

Нужны Linux x86_64 с AVX2, Git, mise, Podman и podman-compose.
Все команды Podman выполняйте одним пользователем без sudo. Для загрузки весов
и сборки образов нужен интернет. Инференс работает без сети и без GPU.

## Обновление и подготовка моделей

Из каталога проекта:

```sh
git switch work
git pull --ff-only origin work
# Только при первой настройке; существующий .env не перезаписывать:
cp -n .env.example .env
mise trust
mise install
mise run models
```

`mise r models` — сокращение той же команды. Она устанавливает только базовые
Python-зависимости через uv и последовательно готовит все четыре провайдера.
Paddle и llama.cpp на хост для загрузки весов не нужны. Можно выбрать один или несколько:

```sh
mise r models lighton_small
mise r models paddle glm_ocr dots_mocr
```

По умолчанию веса и manifests лежат в `./models`. Для другого каталога задайте
`OCR_MODELS_DIR=/полный/путь/models` в `.env`. Этот же путь используется Compose;
в контейнере он доступен как `/models` только для чтения. Подкаталоги VLM
разделены по backend, Paddle хранит свои модели по именам в общем корне.
Повторная подготовка проверяет имеющиеся файлы и загружает отсутствующие.
После изменения квантизации повторите подготовку соответствующего backend.
`OCR_ENABLED_BACKENDS` управляет интерфейсом, а список аргументов `models` — загрузкой.

Каталог должен быть доступен на чтение UID 10001 внутри контейнера. Скрипт
загрузки задаёт umask 022. Для ранее скопированных весов при необходимости:

```sh
chmod -R a+rX models
```

## Перенос весов из прежнего тома

Если веса уже загружены в `document-ocr_models`, их можно перенести без скачивания.
Сначала дождитесь завершения обработки и остановите прежние контейнеры:

```sh
podman compose --profile '*' down
mkdir -p models
podman volume export document-ocr_models --output /tmp/ocr-models.tar
tar -xf /tmp/ocr-models.tar -C models --no-same-owner
chmod -R a+rX models
```

Для нестандартного `OCR_MODELS_DIR` используйте его вместо `models`.
Старый том не удаляется; данные запусков остаются в `document-ocr_data`.

## Запуск и последовательное сравнение

Поднимите API и LightOn Small:

```sh
podman compose up --build -d api
podman compose --profile lighton_small up --build -d lighton-small
```

Откройте http://localhost:8000, выберите LightOn и загрузите PDF.
После завершения обработки переключайте сервисы и распознавайте те же документы:

```sh
podman compose --profile lighton_small stop lighton-small
podman compose --profile paddle up --build -d paddle

# После завершения Paddle:
podman compose --profile paddle stop paddle
podman compose --profile glm_ocr up --build -d glm-ocr

# После завершения GLM:
podman compose --profile glm_ocr stop glm-ocr
podman compose --profile dots_mocr up --build -d dots-mocr
```

Работают только явно выбранные профили. Запуск всех одновременно складывает
потребление CPU и RAM; для сравнения на одной машине запускайте их по очереди.
Результаты документов и отчёты HTML/JSON доступны внутри запусков в интерфейсе.

## Логи и готовность

```sh
podman compose logs -f --tail=100 api
podman compose --profile lighton_small logs -f --tail=100 lighton-small
podman compose --profile paddle logs -f --tail=100 paddle
podman compose --profile glm_ocr logs -f --tail=100 glm-ocr
podman compose --profile dots_mocr logs -f --tail=100 dots-mocr
# Все сервисы:
podman compose --profile '*' logs -f --tail=100
podman ps -a
curl -fsS http://localhost:8000/health
curl -i http://localhost:8000/ready
```

Ctrl+C завершает только просмотр логов. `/health` проверяет API, `/ready` —
наличие готового worker. Проверка настоящего инференса — обработка короткого PDF:
модель загружается в дочернем процессе при получении задания.
При отсутствии весов выполните `mise r models BACKEND` и перезапустите сервис.
Для диагностики dots доступен `compose.debug.yaml`:

```sh
podman compose -f compose.yaml -f compose.debug.yaml --profile dots_mocr up -d --force-recreate dots-mocr
```

## Параметры и ресурсы

Каждый worker в `compose.yaml` содержит собственный список переменных:

| Сервис | Начальная конфигурация |
| --- | --- |
| Paddle | 250 DPI, повтор 350 DPI, MKL-DNN, порог качества 0.75 |
| LightOn Small | Q4_K_M, Q8 vision, transcription, контекст 16384 |
| GLM | Q8_0, задача text, контекст 16384 |
| dots.mocr | Q5_K_M, Q8 vision, контекст 32768 |

VLM рендерят при 150 DPI, лимит входного изображения — 2.5 Мп, KV-cache — q8_0.
Это стартовые настройки; оптимальное время и качество проверяйте на своих PDF.
`OCR_CPU_THREADS=auto` использует доступные потоки с учётом affinity и квот;
`OCR_CPU_RESERVE_CORES=1` оставляет одно физическое ядро системе.
`OCR_CPU_NICE=10` уступает CPU другим процессам при конкуренции и не ограничивает
загрузку свободного процессора. LightOn использует `OCR_LIGHTON_SMALL_THREADS=auto`.
Настройки задания фиксируются API при загрузке, поэтому при изменении настроек
распознавания пересоздайте API вместе с worker и создайте новый запуск.

## Остановка, резервные копии и offline

```sh
podman compose --profile '*' down
mkdir -p backup
podman volume export document-ocr_data --output backup/data.tar
tar -cf backup/models.tar -C models .
```

Остановка сохраняет данные и модели. `down -v` удаляет том с документами и
результатами. Каталог весов на хосте не удаляется Compose.
Для восстановления при остановленных сервисах:

```sh
podman volume create document-ocr_data
podman volume import document-ocr_data backup/data.tar
mkdir -p models
tar -xf backup/models.tar -C models --no-same-owner
```

Для изолированной машины подготовьте модели и соберите нужные образы на машине
с сетью. Перенесите каталог `models`, проект с `.env` и образы через
`podman save` / `podman load`. Запускайте `up -d` без `--build`.
Runtime не скачивает модели; отдельных сервисов `models-*` больше нет.
SQLite и документы держите на локальном диске, не NFS/SMB.
API опубликован на loopback. Для удалённого просмотра:

```sh
ssh -L 8000:127.0.0.1:8000 user@ocr-host
```

Реальная скорость и качество всех моделей на CPU требуют замеров на вашем
оборудовании. Сведения об алгоритмах — [backends.md](backends.md),
о проверках — [validation.md](validation.md).
