"""Portable reports built from saved page timings and runtime metadata."""

import html
import json
from datetime import UTC, datetime

from .catalog import LABELS, VLM_BACKENDS
from .schemas import TERMINAL_STATUSES, DocumentResult

STATUS = {
    "queued": "В очереди",
    "processing": "Распознаётся",
    "pending": "Не начата",
    "cancelling": "Останавливается",
    "cancelled": "Остановлено",
    "completed": "Готово",
    "partial": "Частичный результат",
    "failed": "Ошибка",
}
STAGES = {
    "render_pdf": "Рендер страницы PDF",
    "save_input_image": "Сохранение исходного PNG",
    "recognize": "Предобработка и инференс",
    "save_raw_result": "Сохранение сырого JSON",
    "parse_result": "Разбор текста и таблиц",
    "save_result_image": "Сохранение итогового PNG",
}


def pipeline_description(backend: str) -> list[str]:
    common = [
        "PDF проверяется PyMuPDF: формат, пароль, число страниц. Каждая "
        "страница обрабатывается отдельно.",
        "PyMuPDF рендерит страницу при указанном DPI в RGB без alpha; массив "
        "преобразуется в BGR uint8. "
        "Изображения сверх лимита пикселей отклоняются. Размер и DPI каждого "
        "прохода записываются ниже.",
    ]
    if backend == "paddle":
        common += [
            "PP-StructureV3 на CPU: определение ориентации, анализ макета, "
            "детекция и распознавание "
            "текста, выделение и восстановление структуры таблиц. Выпрямление страницы, формулы, "
            "печати и диаграммы отключены; OCR строк использует batch=8.",
            "Если результат требует проверки, выполняется повтор при retry DPI. Выбирается проход "
            "с лучшей уверенностью без существенной потери текста; ошибки повторов сохраняются.",
        ]
    elif backend in VLM_BACKENDS:
        common += [
            "RGB-изображение уменьшается с сохранением пропорций только при превышении бюджета "
            "VLM-пикселей, кодируется в PNG/base64 и передаётся локальному llama.cpp. Фактические "
            "входные размеры и факт уменьшения записываются в image_preprocessing.",
            "Vision encoder и генерация текста работают на CPU, GPU-offload отключён. Один слот; "
            "temperature=0, seed=42; промпт, контекст, квантизация и лимит токенов "
            "указаны в метаданных. "
            "Проверка качества не запускает повтор с повышенным DPI.",
        ]
    elif backend == "kdl":
        common += [
            "KDL на CPU/FP32: изображение 1036×1036 для Layout Detection; области вырезаются из "
            "исходного рендера, при необходимости поворачиваются и отдельно распознаются. "
            "Обрабатываются текст, таблицы OTSL, формулы и изображения; размер каждого входа "
            "ограничен kdl_max_pixels. Итог объединяется в порядке областей.",
        ]
    elif backend == "falcon":
        common += [
            "Falcon на CPU/FP32: BGR преобразуется в PIL RGB; штатный generate получает "
            "max_dimension=1024, category=plain, temperature=0, seed=42, compile=false. "
            "Предобработка внутри generate входит в общее время инференса; "
            "отдельно она не измеряется.",
        ]
    else:
        common += [
            "Исторический провайдер: точный алгоритм определяется сохранёнными метаданными движка."
        ]
    common += [
        "Сырой ответ сохраняется в JSON. Адаптер извлекает текст/Markdown, блоки и таблицы; "
        "HTML-таблицы очищаются, сохраняются ячейки, ведущие нули и объединения.",
        "Итоговые данные страницы сохраняются в SQLite, изображение выбранного прохода — в PNG. "
        "Ошибка одной страницы не останавливает следующие. Модель загружается "
        "отдельно для каждого документа.",
    ]
    return common


def processing_report(results: list[DocumentResult]) -> dict:
    documents = []
    for result in results:
        document = result.document.model_dump(mode="json")
        pages = []
        for page in result.pages:
            pages.append(
                {
                    "number": page.number,
                    "status": page.status,
                    "seconds": page.seconds
                    if page.status in {"completed", "failed"} and page.seconds > 0
                    else None,
                    "dpi": page.dpi or None,
                    "width": page.width or None,
                    "height": page.height or None,
                    "rotation": page.rotation,
                    "attempts": page.attempts,
                    "passes": page.passes,
                    "metrics": page.metrics,
                    "tables": len(page.tables),
                    "needs_review": page.needs_review,
                    "warnings": page.warnings,
                    "error": page.error,
                }
            )
        documents.append(
            {
                "document": document,
                "provider": LABELS.get(result.document.backend, result.document.backend),
                "engine": result.engine,
                "pipeline": pipeline_description(result.document.backend),
                "pages": pages,
            }
        )
    return {
        "schema_version": "1.0",
        "generated_at": datetime.now(UTC).isoformat(),
        "complete": all(item.document.status in TERMINAL_STATUSES for item in results),
        "notes": [
            "Время страницы — wall time рендера, предобработки, инференса, разбора "
            "и записи файлов. "
            "Ожидание в очереди и загрузка модели в него не входят; загрузка показана отдельно.",
            "CPU seconds и пик суммы RSS измеряются по процессу инференса и "
            "наблюдаемым живым потомкам "
            "каждые 100 мс. Короткие пики могут быть пропущены; общие страницы "
            "памяти могут учитываться "
            "несколько раз. Это оценка, а не точный эксклюзивный пик RAM и не лимит памяти.",
            "null / «Не записано» означает отсутствие измерения. Старые результаты "
            "могут не содержать "
            "этапных таймингов и ресурсов. Для активных документов отчёт является "
            "промежуточным снимком.",
            "Время отдельных внутренних операций модели не выводится как измеренное, если движок "
            "его не предоставил. Сумма этапов может быть меньше общего времени страницы.",
        ],
        "totals": {
            "documents": len(results),
            "pages": sum(len(item.pages) for item in results),
            "processed_pages": sum(item.document.processed_pages for item in results),
            "model_load_seconds": sum(item.document.model_load_seconds for item in results),
            "processing_seconds": sum(item.document.processing_seconds for item in results),
        },
        "documents": documents,
        "upload_errors": [],
    }


def html_report(report: dict) -> str:
    def esc(value):
        return html.escape(str(value))

    def number(value):
        return "Не записано" if value is None else f"{value:.3f}"

    def data(value):
        return "<pre>" + esc(json.dumps(value, ensure_ascii=False, indent=2)) + "</pre>"

    parts = [
        '<!doctype html><html lang="ru"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        '<meta http-equiv="Content-Security-Policy" '
        "content=\"default-src 'none'; style-src 'unsafe-inline'\">",
        "<title>Отчёт обработки PDF</title><style>body{font:15px/1.6 system-ui;max-width:1200px;"
        "margin:2rem auto;padding:0 1rem;color:#203044}table{border-collapse:collapse;width:100%;"
        "margin:1rem 0}td,th{border:1px solid "
        "#ccd5e0;padding:.5rem;text-align:left;vertical-align:top}"
        "th{background:#f0f4fa}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f3f5f8;"
        "padding:1rem}article{border-top:2px solid #ccd5e0;margin-top:2rem}.scroll{overflow-x:auto}"
        "@media print{details{display:block}table{font-size:11px}}</style></head><body>",
        "<h1>Отчёт обработки PDF</h1>",
        f"<p>Сформирован: {esc(report['generated_at'])} (UTC). "
        + (
            "Все задания завершены."
            if report["complete"]
            else "Промежуточный отчёт: есть незавершённые задания."
        )
        + "</p>",
        "<h2>Сводка</h2>",
        f"<p>Документов в очереди: {report['totals']['documents']}. "
        f"Страниц: {report['totals']['pages']}, обработано: {report['totals']['processed_pages']}. "
        f"Ошибок загрузки: {len(report['upload_errors'])}.</p>",
        "<p>Суммарное время обработки документов: "
        f"{number(report['totals']['processing_seconds'])} с; "
        f"суммарная загрузка моделей: {number(report['totals']['model_load_seconds'])} с. "
        "Эти суммы не являются общим wall time пакета и не включают ожидание в очереди.</p>",
        "<ul>" + "".join("<li>" + esc(note) + "</li>" for note in report["notes"]) + "</ul>",
    ]
    if report["upload_errors"]:
        parts.append(
            "<h2>Не удалось загрузить</h2>"
            "<p>Ошибки загрузки и недоступные документы выбранного запуска.</p>"
        )
        for failure in report["upload_errors"]:
            parts.append(
                "<p><strong>"
                + esc(failure["filename"])
                + "</strong>: "
                + esc(failure["error"])
                + "</p>"
            )
    for item in report["documents"]:
        doc = item["document"]
        engine = item["engine"]
        policy = engine.get("cpu_policy", {})
        host = engine.get("host", {})
        quota = policy.get("quota")
        quota_text = str(quota) if quota is not None else "не записана / без ограничения"
        resources = (
            f"CPU: {host.get('cpu_model') or 'модель CPU не записана'}; "
            f"потоков: {policy.get('threads', engine.get('cpu_threads', 'не записано'))}; "
            f"nice: {policy.get('nice', 'не записано')}; "
            f"CPU affinity: {policy.get('cpu_ids', engine.get('cpu_affinity', 'не записано'))}; "
            f"квота CPU: {quota_text}."
        )
        parts += [
            "<article>",
            f"<h2>{esc(doc['filename'])}</h2>",
            f"<p>Провайдер: <strong>{esc(item['provider'])}</strong> · "
            f"{esc(STATUS.get(doc['status'], doc['status']))} · ID: {esc(doc['id'])}</p>",
            f"<p>Загрузка модели: {number(doc['model_load_seconds'])} с. "
            f"Обработка документа: {number(doc['processing_seconds'])} с.</p>",
            "<p><strong>Ресурсы:</strong> " + esc(resources) + "</p>",
            "<h3>Пайплайн обработки</h3><ol>"
            + "".join("<li>" + esc(step) + "</li>" for step in item["pipeline"])
            + "</ol>",
            "<details><summary>Модель, настройки и выделенные ресурсы</summary>",
            data(item["engine"]),
            "</details>",
            '<div class="scroll"><table><thead><tr><th>Страница</th><th>Статус</th>'
            "<th>Время, с</th>"
            "<th>DPI / размер</th><th>CPU, с</th><th>Пик суммы RSS, МиБ</th></tr></thead><tbody>",
        ]
        for page in item["pages"]:
            resources = page["metrics"].get("resources", {})
            memory = resources.get("sampled_tree_rss_peak_bytes")
            dimensions = (
                f"{page['dpi']} dpi · {page['width']}×{page['height']}"
                if page["dpi"]
                else "Не записано"
            )
            parts.append(
                f"<tr><td>{page['number']}</td>"
                f"<td>{esc(STATUS.get(page['status'], page['status']))}</td>"
                f"<td>{number(page['seconds'])}</td><td>{esc(dimensions)}</td>"
                f"<td>{number(resources.get('cpu_seconds'))}</td>"
                f"<td>{number(memory / 1024**2 if memory is not None else None)}</td></tr>"
            )
        parts.append("</tbody></table></div>")
        for page in item["pages"]:
            parts += [f"<h3>Страница {page['number']}: этапы и параметры</h3>"]
            for message in [page["error"], *page["warnings"]]:
                if message:
                    parts.append("<p>" + esc(message) + "</p>")
            if not page["passes"]:
                parts.append("<p>Этапные измерения не записаны.</p>")
            for index, attempt in enumerate(page["passes"], 1):
                parts.append(f"<h4>Проход {index} · {esc(attempt.get('dpi'))} dpi</h4>")
                if attempt.get("error"):
                    parts.append("<p>" + esc(attempt["error"]) + "</p>")
                for stage in attempt.get("stages", []):
                    parts.append(
                        "<p><strong>"
                        + esc(STAGES.get(stage["stage"], stage["stage"]))
                        + "</strong>: "
                        + number(stage["seconds"])
                        + " с · "
                        + esc(stage["status"])
                        + "</p>"
                    )
                    parts.append(data(stage.get("details", {})))
                if attempt.get("metrics"):
                    parts += [
                        "<details><summary>Метрики модели и предобработка изображения</summary>",
                        data(attempt["metrics"]),
                        "</details>",
                    ]
            parts += [
                "<details><summary>Все метрики страницы, финализация и ресурсы</summary>",
                data(page["metrics"]),
                "</details>",
            ]
        parts.append("</article>")
    return "\n".join(parts) + "</body></html>"
