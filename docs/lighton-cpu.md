# Настройка LightOnOCR на CPU

Для LightOnOCR-3-0.8B и 1B доступны decoder Q8_0 и Q5_K_M. Vision encoder
каждой модели остаётся своим Q8_0; DPI, размер изображения и промпт `grounding`
сохраняются. Q8_0 остаётся значением по умолчанию. Q5 может изменить качество
и не гарантирует ускорение на каждом CPU.

## Потоки на Intel Core i9-11900

У процессора 8 физических ядер и 16 логических потоков. Значение 20 создаёт
больше потоков, чем логических CPU. Для первого сравнения задайте в `.env`:

```dotenv
OCR_CPU_THREADS=8
OCR_CPU_THREADS_BATCH=16
OCR_LIGHTON_SMALL_QUANTIZATION=Q8_0
```

`OCR_CPU_THREADS` задаёт `--threads`: генерацию текста и визуальный энкодер.
`OCR_CPU_THREADS_BATCH` задаёт только `--threads-batch`: пакетную обработку
входных токенов decoder. В закреплённом llama.cpp число потоков визуального
энкодера берётся из `--threads`, поэтому отдельной настройки mmproj здесь нет.
Значение batch `0` наследует `OCR_CPU_THREADS`, сохраняя прежнее поведение.

Остановите текущий прогон кнопкой в клиенте, затем обновите и пересоздайте
API и worker, чтобы оба получили новые настройки:

```sh
git pull --ff-only origin work
podman compose --profile lighton_small up --build --force-recreate -d
```

Загрузите PDF заново: параметры фиксируются при загрузке. Старые задания
сохраняют свои настройки, включая прежние 20 потоков. В JSON нового задания
проверьте `engine.cpu_threads=8`, `engine.cpu_threads_batch=16`.

Сравните тот же документ при 8, 12 и 16 потоках генерации, сохраняя batch=16.
Для изменения только `.env` достаточно `up --force-recreate -d` без `--build`.
Сопоставляйте `eval time`, токены/с, `prompt eval time` и время всей страницы:
ускорение генерации может сопровождаться замедлением визуального энкодера.
Не меняйте одновременно количество потоков и квантизацию.

## Decoder Q5_K_M

После замера Q8 измените только:

```dotenv
OCR_LIGHTON_SMALL_QUANTIZATION=Q5_K_M
```

```sh
podman compose --profile lighton_small up --force-recreate -d
podman compose --profile lighton_small logs -f models-lighton-small
```

Подготовка скачает готовый Q5 decoder (~578 МБ) с проверкой pinned SHA256;
готовый Q8 vision encoder используется повторно. Локальная конвертация
и переквантизация Q8 не выполняются. Затем загрузите тот же PDF заново;
в клиенте и JSON будет указано `Q5_K_M`.

Для LightOnOCR-3-1B используются `OCR_LIGHTON_QUANTIZATION=Q5_K_M`, профиль
`lighton`, worker `lighton` и загрузчик `models-lighton`. Q5 decoder ~444 МБ.
Значения выбираются независимо; одновременно запускать обе модели не нужно.

Q8 и Q5 используют отдельные manifest-файлы. Возврат к Q8 выполняется через
значение `Q8_0`, пересоздание сервисов и новую загрузку PDF. Сравнивайте не
только скорость, но и цифры, ведущие нули, пустые и объединённые ячейки.
Скорость и качество настоящих Q5-весов в среде разработки не измерены.

Источники decoder и SHA256:

- [LightOnOCR-3-0.8B Q5_K_M](https://huggingface.co/prithivMLmods/LightOnOCR-3-0.8B-GGUF/blob/main/LightOnOCR-3-0.8B.Q5_K_M.gguf),
  revision `414757bdb78fd07c947a1e83c2b8d7e9a8b73583`, SHA256
  `fc87991ce656f03c23f45662c92a58048bd0620798ffd65471707a07e12c14ad`.
- [LightOnOCR-3-1B Q5_K_M](https://huggingface.co/prithivMLmods/LightOnOCR-3-1B-GGUF/blob/main/LightOnOCR-3-1B.Q5_K_M.gguf),
  revision `ff5c331ec2cfee9de1c8054c21ccf88b8c5b0faa`, SHA256
  `6c9ca18e0531f9f5277e3cd982b84ba83e5f476d2ee239a2288f82413f2ce355`.
