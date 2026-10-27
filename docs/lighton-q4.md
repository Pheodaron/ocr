# LightOnOCR-3-0.8B: Q4, автоматический бюджет CPU, транскрипция

Профиль `lighton_small` по умолчанию использует новый эксперимент:

```dotenv
OCR_LIGHTON_SMALL_QUANTIZATION=Q4_K_M
OCR_LIGHTON_SMALL_TASK=transcription
OCR_LIGHTON_SMALL_THREADS=auto
OCR_LIGHTON_SMALL_PIN_THREADS=true
```

## Веса

Decoder и vision encoder берутся из одного набора
[prithivMLmods/LightOnOCR-3-0.8B-GGUF](https://huggingface.co/prithivMLmods/LightOnOCR-3-0.8B-GGUF),
revision `414757bdb78fd07c947a1e83c2b8d7e9a8b73583`:

- `LightOnOCR-3-0.8B.Q4_K_M.gguf`, ~529 МБ, SHA256
  `290ba9b31716a00dc1cbe1555a2776109f21ae3f4f5c177a2b543c9bc3eab12a`.
- `LightOnOCR-3-0.8B.mmproj-q8_0.gguf`, SHA256
  `a8b9b3b01b6c780747b48e0807ef5b753c54f27aa4c23962c3c7d5ace2ed0233`.

Используется прежний Q8 vision encoder без дополнительного сжатия.
Подготовка проверяет SHA256; готовый vision используется повторно.
Q4 и прежний Q8 decoder используют разные файлы и manifest.
Runtime работает без сети и только проверяет локальные файлы.
Совместимость пары, скорость и качество требуют реального прогона;
в среде разработки веса недоступны.

## CPU

`OCR_LIGHTON_SMALL_THREADS=auto` использует общий механизм выбора CPU:
разрешённые логические потоки, SMT, видимые квоты cgroup и резерв
`OCR_CPU_RESERVE_CORES=1` для системы. Ручное число потоков является верхним
пределом; прежнего ограничения 1–8 больше нет. `OCR_CPU_THREADS` задаёт
бюджет остальных моделей.

Маска исключает все SMT-потоки зарезервированного физического ядра. Оставшиеся
физические ядра заполняются перед добавлением их SMT-потоков. Дочерний процесс
инференса получает `nice=10`; OMP/MKL/OpenBLAS/NumExpr/Rayon используют тот же
бюджет. Decoder и обработка изображения получают одинаковые `--threads` и
`--threads-batch`.

`OCR_LIGHTON_SMALL_PIN_THREADS=false` отключает дополнительные CPU-mask флаги
llama.cpp для этой модели; общий резерв процесса при этом сохраняется.
Фактический бюджет записывается в `engine.cpu_threads`, `engine.cpu_affinity`
и `engine.cpu_policy`. Подробнее: [настройки CPU](../README.md#локальная-разработка-с-mise).

## Промпт и результат

`transcription` отправляет изображение без текстовой части сообщения,
что соответствует [пустому промпту из карточки LightOn](https://huggingface.co/lightonai/LightOnOCR-3-0.8B#prompt-modes).
Модель возвращает текст и Markdown/HTML; адаптер сохраняет таблицы, ведущие
нули и объединения HTML-ячеек. Координаты без grounding не выдумываются.
Для возврата меток и координат задайте `OCR_LIGHTON_SMALL_TASK=grounding`.
Режим виден в клиенте и JSON. Настройки 1B здесь не меняются.

DPI, размер страницы, контекст и лимит ответа сохраняются. Улучшение токенов/с
и времени страницы не гарантировано. Официальная оценка ~25% дополнительных
токенов grounding относится к замеру 4B; это не замер 0.8B или вашего CPU.

## Запуск

Остановите активное задание кнопкой в клиенте, затем:

```sh
git pull --ff-only origin work
mise r models lighton_small
podman compose --profile lighton_small up --build --force-recreate -d
podman compose --profile lighton_small logs -f lighton-small
```

Загрузите PDF заново. Старые задания без новых полей сохраняют Q8 decoder
и grounding; к выполняемым заданиям LightOn 0.8B применяется новый лимит CPU.
Для повторного сравнения с Q8 задайте `OCR_LIGHTON_SMALL_QUANTIZATION=Q8_0`,
пересоздайте сервисы и загрузите PDF заново. Сравнивайте одинаковые страницы:
сначала качество таблиц и цифр, затем число токенов, токены/с и полное время.
