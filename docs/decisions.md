> История решений. Текущие провайдеры: Paddle, dots.mocr, GLM-OCR и LightOn 0.8B.
> Ранее добавленные модели больше не поддерживаются; актуальная конфигурация — в README.

# Согласованный первый срез

- Linux x86_64, только CPU; русский и английский.
- Вход: один PDF на задание, загрузка через HTTP endpoint.
- Результат: текст, таблицы, координаты, оценки OCR и предупреждения; JSON,
  Markdown и HTML. Qwen3 и заполнение XLSX отложены.
- Простой клиент: одна веб-страница, без отдельного frontend toolchain.
- API асинхронный: загрузка → ID → опрос статуса → получение результата.
- Доступ только через localhost, без авторизации в первой версии.
- Ошибка одной страницы сохраняется, обработка следующих продолжается.
- В первом срезе один OCR worker; API работает отдельным процессом. SQLite хранит очередь и
  результаты страниц. Данные сохраняются между перезапусками.
- Запуск через Podman Compose, альтернативно через mise/uv.
- Интернет нужен для сборки и первоначальной подготовки моделей. Runtime OCR
  использует только локальные веса, у worker отключена сеть.
- Подготовленные модели можно перенести в изолированную систему отдельно.

Начальные настраиваемые ограничения: PDF до 64 МиБ, 300 страниц, 30 млн пикселей
на страницу, 250 dpi и один повтор на 350 dpi. Файлы и результаты сохраняются
до ручного удаления каталога/тома; автоматической очистки в этой версии нет.

Решения уточнялись по навыку Matt Pocock grill-me / grilling:
https://github.com/mattpocock/skills/blob/main/skills/productivity/grilling/SKILL.md

## Расширение для сравнения конфигураций

По последующим указаниям добавлены dots.mocr (Q5_K_M и более точные варианты),
GLM-OCR Q8_0, Chandra OCR 2, LightOnOCR-3-1B и KDL-Frontier-Parser-nano.
Один PDF можно отправить нескольким движкам. Каждый прогон получает отдельный ID,
снимок настроек и результат. Каждая модель запускается отдельным сервисом Compose в своём профиле; задания
выключенных моделей ждут. Внутри сервиса очередь последовательная; отдельный
процесс на документ освобождает память модели после прогона. VLM заменяет
распознавание страницы; этап Qwen3 по бизнес-нормализации не добавлялся.
KDL реализован через исходный CPU checkpoint и двухэтапную обработку областей,
остальные новые варианты используют локальный multimodal llama.cpp.


## Independent CPU comparison services

Every model has its own Compose service/profile and backend-filtered SQLite queue.
A missing or stopped worker leaves only its own jobs waiting. Each document runs
in a child process so loaded models release RAM when the document finishes.
Jina and Nemotron support is currently removed.
Surya uses the original model's official F16 GGUF, as explicitly selected by the user.
Falcon requires a CPU port of upstream Triton operations; real-weight validation
is pending because Hugging Face downloads are blocked in this environment.
