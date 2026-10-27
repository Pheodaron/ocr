LABELS = {
    "paddle": "Paddle PP-StructureV3",
    "dots_mocr": "dots.mocr",
    "glm_ocr": "GLM-OCR",
    "lighton_small": "LightOnOCR-3-0.8B",
}
VLM_BACKENDS = {"dots_mocr", "glm_ocr", "lighton_small"}

# Display-only metadata for saved results and compatibility with older .env lists.
ARCHIVED_LABELS = {
    "nemotron": "Nemotron 3.5 Parse",
    "jina": "jina-ocr-v1",
    "surya": "Surya OCR 2",
    "falcon": "Falcon-OCR 1.5",
    "chandra": "Chandra OCR 2",
    "kdl": "KDL-Frontier-Parser-nano",
    "lighton": "LightOnOCR-3-1B",
}
REMOVED_BACKENDS = frozenset(ARCHIVED_LABELS)


def backend_label(backend: str) -> str:
    return LABELS.get(backend, ARCHIVED_LABELS.get(backend, backend))
