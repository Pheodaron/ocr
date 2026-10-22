# LightOnOCR-3-0.8B: Q4, восемь физических ядер, транскрипция

Профиль `lighton_small` по умолчанию использует новый эксперимент:

```dotenv
OCR_LIGHTON_SMALL_QUANTIZATION=Q4_K_M
OCR_LIGHTON_SMALL_TASK=transcription
OCR_LIGHTON_SMALL_THREADS=8
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

Для `lighton_small` отдельный лимит `OCR_LIGHTON_SMALL_THREADS` равен 8,
допустимы значения 1–8. Он задаёт и `--threads`, и `--threads-batch`, включая
число потоков vision encoder. Глобальное `OCR_CPU_THREADS=20` этот лимит
не переопределяет. Остальные backend используют прежние настройки.

При pinning выбирается по одному разрешённому логическому CPU на каждую
пару `(physical_package_id, core_id)` через Linux sysfs с учётом
`sched_getaffinity`. CPU с общим физическим ядром повторно не выбираются.
Процесс llama-server и его потоки наследуют полученную маску. Для decoder
дополнительно задаются `--cpu-mask`, `--cpu-strict 1` и соответствующие batch
параметры. В закреплённой версии нет `--pin-threads yes`.

В окружении llama-server устанавливаются OMP/MKL/OpenBLAS/NumExpr/Rayon
лимиты. Если контейнеру доступны менее восьми физических ядер, используется
меньшее число. Если topology недоступна, остаётся лимит потоков, а в журнале
будет предупреждение об отсутствии физической привязки. Отключение pinning:
`OCR_LIGHTON_SMALL_PIN_THREADS=false`.

Фактические потоки и CPU записываются в `engine.cpu_threads` и
`engine.cpu_affinity`. Это ограничивает использование Hyper-Threading данным
процессом, не отключая его в BIOS или во всей системе.

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
podman compose --profile lighton_small up --build --force-recreate -d
podman compose --profile lighton_small logs -f models-lighton-small lighton-small
```

Загрузите PDF заново. Старые задания без новых полей сохраняют Q8 decoder
и grounding; к выполняемым заданиям LightOn 0.8B применяется новый лимит CPU.
Для повторного сравнения с Q8 задайте `OCR_LIGHTON_SMALL_QUANTIZATION=Q8_0`,
пересоздайте сервисы и загрузите PDF заново. Сравнивайте одинаковые страницы:
сначала качество таблиц и цифр, затем число токенов, токены/с и полное время.
