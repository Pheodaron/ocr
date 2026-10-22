import hashlib
import json

import pytest

from ocr_service.backends import configuration_details
from ocr_service.vlm import server_command
from ocr_service.vlm_models import artifacts, prepare_vlm, verify_vlm


@pytest.mark.parametrize(
    "backend,key,name,revision,q5_hash",
    [
        (
            "lighton_small",
            "lighton_small_quantization",
            "LightOnOCR-3-0.8B",
            "414757bdb78fd07c947a1e83c2b8d7e9a8b73583",
            "fc87991ce656f03c23f45662c92a58048bd0620798ffd65471707a07e12c14ad",
        ),
        (
            "lighton",
            "lighton_quantization",
            "LightOnOCR-3-1B",
            "ff5c331ec2cfee9de1c8054c21ccf88b8c5b0faa",
            "6c9ca18e0531f9f5277e3cd982b84ba83e5f476d2ee239a2288f82413f2ce355",
        ),
    ],
)
def test_lighton_q5_keeps_own_projector_and_pinned_upstream_integrity(
    settings, backend, key, name, revision, q5_hash
):
    baseline = artifacts(backend, settings)
    selected = settings.model_copy(update={key: "Q5_K_M"})
    spec = artifacts(backend, selected)
    assert spec["model"] == f"{name}.Q5_K_M.gguf"
    assert spec["projector"] == baseline["projector"] == f"{name}.mmproj-q8_0.gguf"
    assert spec["revision"] == revision == baseline["revision"]
    assert spec["files"][spec["model"]] == q5_hash
    assert len(bytes.fromhex(q5_hash)) == 32
    assert spec["files"][spec["projector"]] == baseline["files"][baseline["projector"]]
    assert configuration_details(backend, selected.recognition_config())["quantization"] == "Q5_K_M"


@pytest.mark.parametrize(
    "backend,key",
    [("lighton_small", "lighton_small_quantization"), ("lighton", "lighton_quantization")],
)
def test_lighton_q5_preparation_keeps_q8_results_reproducible(settings, monkeypatch, backend, key):
    from ocr_service import vlm_models

    get_spec = artifacts
    fixtures = {}

    def spec_with_fixture_hashes(backend, settings):
        spec = get_spec(backend, settings)
        for filename in spec["files"]:
            fixtures.setdefault(filename, b"GGUFfixture-" + filename.encode())
            spec["files"][filename] = hashlib.sha256(fixtures[filename]).hexdigest()
        return spec

    monkeypatch.setattr(vlm_models, "artifacts", spec_with_fixture_hashes)
    directory = settings.models_dir / backend
    directory.mkdir(parents=True)
    baseline = spec_with_fixture_hashes(backend, settings)
    for filename in baseline["files"]:
        (directory / filename).write_bytes(fixtures[filename])
    q8 = prepare_vlm(backend, settings)
    q8_manifest = (directory / "manifest-Q8_0.json").read_bytes()
    selected = settings.model_copy(update={key: "Q5_K_M"})
    q5_spec = spec_with_fixture_hashes(backend, selected)
    downloaded = []

    def download(url, **kwargs):
        import io

        downloaded.append(url)
        return io.BytesIO(fixtures[q5_spec["model"]])

    monkeypatch.setattr(vlm_models.urllib.request, "urlopen", download)
    q5 = prepare_vlm(backend, selected, download=True)
    assert downloaded == [
        f"https://huggingface.co/{q5_spec['repo']}/resolve/{q5_spec['revision']}/{q5_spec['model']}"
    ]
    assert (directory / "manifest-Q8_0.json").read_bytes() == q8_manifest
    assert json.loads((directory / "manifest-Q5_K_M.json").read_text()) == q5
    assert verify_vlm(backend, selected) == q5
    assert verify_vlm(backend, settings) == q8
    assert configuration_details(backend, settings.recognition_config())["quantization"] == "Q8_0"


def test_uploaded_lighton_configuration_survives_worker_changes(client, pdf_bytes, settings):
    job = client.post(
        "/api/documents", files={"file": ("test.pdf", pdf_bytes)}, data={"backend": "lighton_small"}
    ).json()
    db = client.app.state.db
    queued = db.claim_job()
    original = json.loads(queued["config_json"])
    changed = settings.model_copy(
        update={"cpu_threads": 8, "cpu_threads_batch": 16, "lighton_small_quantization": "Q5_K_M"}
    )
    restored = changed.restore_recognition_config(original)
    assert restored.lighton_small_quantization == "Q8_0"
    assert restored.batch_threads == settings.cpu_threads
    assert (
        client.get(f"/api/documents/{job['id']}").json()["configuration"]["quantization"] == "Q8_0"
    )
    new_snapshot = changed.recognition_config()
    restored_new = settings.restore_recognition_config(new_snapshot)
    assert restored_new.lighton_small_quantization == "Q5_K_M"
    assert restored_new.cpu_threads == 8 and restored_new.batch_threads == 16
    assert restored_new.vlm_dpi == settings.vlm_dpi
    assert restored_new.vlm_max_pixels == settings.vlm_max_pixels


def test_legacy_jobs_keep_inherited_threads_and_q8_after_worker_changes(settings):
    legacy = {"dpi": settings.dpi, "cpu_threads": 20}
    changed = settings.model_copy(
        update={"cpu_threads": 8, "cpu_threads_batch": 16, "lighton_small_quantization": "Q5_K_M"}
    )
    restored = changed.restore_recognition_config(legacy)
    assert restored.cpu_threads == restored.batch_threads == 20
    assert restored.lighton_small_quantization == restored.lighton_quantization == "Q8_0"
    command = server_command(restored, "lighton_small", artifacts("lighton_small", restored))
    assert command[command.index("--threads") + 1] == "20"
    assert command[command.index("--threads-batch") + 1] == "20"
