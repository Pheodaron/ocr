import hashlib
import json
import os
import subprocess
import sys
from functools import partial

import pytest

from ocr_service.backends import configuration_details
from ocr_service.models import ModelsUnavailable
from ocr_service.vlm import parent_death_signal
from ocr_service.vlm_adapter import adapt_vlm
from ocr_service.vlm_models import artifacts, prepare_vlm, verify_vlm


def test_native_child_inherits_physical_affinity_without_changing_parent():
    available = os.sched_getaffinity(0)
    selected = [min(available)]
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import os,json; print(json.dumps(sorted(os.sched_getaffinity(0))))",
        ],
        preexec_fn=partial(parent_death_signal, cpu_ids=selected),
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(result.stdout) == selected
    assert os.sched_getaffinity(0) == available


def test_q4_spec_has_pinned_decoder_and_existing_q8_vision_from_same_source(settings):
    spec = artifacts("lighton_small", settings)
    baseline = artifacts(
        "lighton_small", settings.model_copy(update={"lighton_small_quantization": "Q8_0"})
    )
    assert spec["repo"] == baseline["repo"] == "prithivMLmods/LightOnOCR-3-0.8B-GGUF"
    assert spec["revision"] == baseline["revision"] == "414757bdb78fd07c947a1e83c2b8d7e9a8b73583"
    assert spec["model"] == "LightOnOCR-3-0.8B.Q4_K_M.gguf"
    assert (
        spec["files"][spec["model"]]
        == "290ba9b31716a00dc1cbe1555a2776109f21ae3f4f5c177a2b543c9bc3eab12a"
    )
    assert spec["projector"] == baseline["projector"]
    assert spec["files"][spec["projector"]] == baseline["files"][baseline["projector"]]


def test_model_preparation_pins_files_reuses_cache_and_checks_manifest(settings, monkeypatch):
    import io

    from ocr_service import vlm_models

    spec = artifacts("lighton_small", settings)
    data = {filename: b"GGUFfixture-" + filename.encode() for filename in spec["files"]}
    spec["files"] = {filename: hashlib.sha256(body).hexdigest() for filename, body in data.items()}
    monkeypatch.setattr(vlm_models, "artifacts", lambda *_: spec)
    requested = []

    def download(url, **_kwargs):
        requested.append(url)
        return io.BytesIO(data[url.rsplit("/", 1)[-1]])

    monkeypatch.setattr(vlm_models.urllib.request, "urlopen", download)
    manifest = prepare_vlm("lighton_small", settings, download=True)
    assert requested == [
        f"https://huggingface.co/{spec['repo']}/resolve/{spec['revision']}/{spec['model']}",
        f"https://huggingface.co/prithivMLmods/LightOnOCR-3-0.8B-GGUF/resolve/414757bdb78fd07c947a1e83c2b8d7e9a8b73583/{spec['projector']}",
    ]
    assert verify_vlm("lighton_small", settings) == manifest
    prepare_vlm("lighton_small", settings)
    assert len(requested) == 2
    path = settings.models_dir / "lighton_small/manifest-Q4_K_M.json"
    manifest = json.loads(path.read_text())
    manifest["revision"] = "wrong"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ModelsUnavailable, match="manifest"):
        verify_vlm("lighton_small", settings)


def test_new_jobs_use_q4_transcription_and_old_jobs_keep_q8_grounding(client, pdf_bytes, settings):
    job = client.post(
        "/api/documents", files={"file": ("test.pdf", pdf_bytes)}, data={"backend": "lighton_small"}
    ).json()
    assert job["configuration"]["quantization"] == "Q4_K_M"
    assert job["configuration"]["task"] == "transcription"
    snapshot = json.loads(client.app.state.db.claim_job()["config_json"])
    assert snapshot["lighton_small_threads"] == "auto" and snapshot["lighton_small_pin_threads"]
    legacy = {"cpu_threads": 20, "dpi": settings.dpi}
    restored = settings.restore_recognition_config(legacy)
    spec = artifacts("lighton_small", restored)
    assert spec["quantization"] == "Q8_0"
    assert restored.lighton_small_task == "grounding"
    assert configuration_details("lighton_small", legacy)["task"] == "grounding"
    assert restored.lighton_small_threads == "auto"


def test_transcription_preserves_html_and_markdown_tables_without_fabricating_boxes():
    for content in (
        "<table><tr><td>00125</td><td>150 кВт</td></tr></table>",
        "| № | Мощность |\n|---|---|\n| 00125 | 150 кВт |",
    ):
        page = adapt_vlm(
            {
                "ocr_backend": "lighton_small",
                "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
            },
            1,
            150,
            1241,
            1754,
        )
        assert page.tables[0].cells[-2].text == "00125"
        assert page.tables[0].cells[-1].text == "150 кВт"
        assert page.blocks == []
        assert page.tables[0].bbox is None
