from ocr_service.vlm_adapter import adapt_vlm


def test_markdown_tables_preserve_zeros_empty_cells_and_escaped_pipes():
    raw = {
        "ocr_backend": "glm_ocr",
        "choices": [
            {
                "finish_reason": "stop",
                "message": {"content": "| № | Значение | Пусто |\n|---|---|---|\n|00125|a\\|b||"},
            }
        ],
    }
    page = adapt_vlm(raw, 1, 250, 100, 100)
    assert [cell.text for cell in page.tables[0].cells] == [
        "№",
        "Значение",
        "Пусто",
        "00125",
        "a|b",
        "",
    ]


def test_every_catalog_model_has_its_own_compose_service_and_preparation():
    from pathlib import Path

    import yaml

    from ocr_service.catalog import LABELS

    compose = yaml.safe_load((Path(__file__).parents[1] / "compose.yaml").read_text())
    services = compose["services"]
    for backend in LABELS:
        workers = [
            (name, value)
            for name, value in services.items()
            if value.get("environment", {}).get("OCR_WORKER_BACKEND") == backend
        ]
        assert len(workers) == 1
        name, worker = workers[0]
        assert worker["profiles"] == [backend]
        assert worker["network_mode"] == "none"
        assert services[f"models-{name}"]["environment"]["OCR_ENABLED_BACKENDS"] == backend
