import ast

import pytest

from ocr_service.falcon import patch_model_source
from ocr_service.hf_models import prepare_hf, verify_hf
from ocr_service.models import ModelsUnavailable
from ocr_service.vlm_adapter import adapt_vlm


def test_falcon_cpu_attention_matches_sdpa_for_image_prefix_and_decode():
    torch = pytest.importorskip("torch")
    from ocr_service.falcon_cpu import (
        CPUMask,
        cpu_attention,
        create_batch_attention_mask,
        offset_mask_mod,
    )

    # Left pad, two bidirectional image tokens, image end, text, then unused slots.
    tokens = torch.tensor([[0, 2, 4, 3, 7, 0, 0]])
    mask = create_batch_attention_mask(
        tokens,
        pad_token_id=0,
        eos_token_id=9,
        soi_token_id=2,
        eoi_token_id=3,
    )
    assert mask.mask_mod(0, 0, 1, 2)  # vision prefix sees a future image token
    assert not mask.mask_mod(0, 0, 4, 0)  # never sees left padding
    assert not mask.mask_mod(0, 0, 4, 5)  # text is causal
    torch.manual_seed(42)
    q, k, v = [torch.randn(1, 2, 7, 4) for _ in range(3)]
    positions = torch.arange(7)
    allowed = mask.mask_mod(0, 0, positions[:, None], positions[None, :])
    expected = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=allowed)
    actual, aux = cpu_attention(q, k, v, block_mask=mask, return_aux=True)
    torch.testing.assert_close(actual, expected)
    assert aux.lse.shape == (1, 2, 7)
    decode_mask = CPUMask(offset_mask_mod(mask.mask_mod, 5))
    decoded = cpu_attention(q[:, :, 5:6], k[:, :, :6], v[:, :, :6], block_mask=decode_mask)
    torch.testing.assert_close(decoded, actual[:, :, 5:6])


def test_falcon_source_patch_removes_gpu_kernels_and_preserves_gate_math():
    torch = pytest.importorskip("torch")
    source = """import torch.nn.functional as F
import triton
import triton.language as tl
from torch.nn.attention.flex_attention import AuxRequest, BlockMask
@triton.jit
def _squared_relu_gate_kernel(x): pass
def squared_relu_gate(packed, hidden_dim): return triton.invalid(packed)
class Model:
    def _load_layout_model(self):
        import torchvision
"""
    patched = patch_model_source(source)
    assert "triton" not in patched and "torchvision" not in patched
    namespace = {}
    exec(compile(ast.parse(patched), "<cpu-patch-fixture>", "exec"), namespace)
    packed = torch.tensor([[-2.0, 3.0, 2.0, 5.0]])
    assert namespace["squared_relu_gate"](packed, 2).tolist() == [[0.0, 20.0]]
    with pytest.raises(ValueError, match="differs"):
        patch_model_source("pass")


def test_pinned_hf_snapshot_checks_code_as_well_as_weights(settings):
    directory = settings.models_dir / "falcon"
    directory.mkdir(parents=True)
    for name in ("config.json", "tokenizer_config.json", "model.safetensors", "modeling.py"):
        (directory / name).write_text("fixture")
    manifest = prepare_hf("falcon", settings)
    assert manifest == verify_hf("falcon", settings)
    (directory / "modeling.py").write_text("changed code")
    with pytest.raises(ModelsUnavailable, match="SHA256"):
        verify_hf("falcon", settings)


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
