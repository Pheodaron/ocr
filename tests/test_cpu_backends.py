import ast
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
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
    directory = settings.models_dir / "nemotron"
    directory.mkdir(parents=True)
    for name in ("config.json", "tokenizer_config.json", "model.safetensors", "modeling.py"):
        (directory / name).write_text("fixture")
    manifest = prepare_hf("nemotron", settings)
    assert manifest == verify_hf("nemotron", settings)
    (directory / "modeling.py").write_text("changed code")
    with pytest.raises(ModelsUnavailable, match="SHA256"):
        verify_hf("nemotron", settings)


def test_nemotron_cpu_contract_preserves_indices_and_upstream_postprocessing(settings, monkeypatch):
    torch = pytest.importorskip("torch")
    from ocr_service import nemotron

    class Batch(dict):
        def to(self, device):
            assert device == "cpu"
            return self

    class Processor:
        def __call__(self, **kwargs):
            assert kwargs["text"] == nemotron.PROMPT
            assert kwargs["images"][0].size == (30, 20)
            return Batch(input_ids=torch.tensor([[1, 2]]), pixel_values=torch.ones(1).half())

        def batch_decode(self, output, **kwargs):
            return ["serialized fixture"]

    model = Mock()
    model.to.return_value.eval.return_value = model
    model.generate.return_value = torch.tensor([[1, 2, 3]])
    transformers = SimpleNamespace(
        AutoModel=Mock(from_pretrained=Mock(return_value=model)),
        AutoProcessor=Mock(from_pretrained=Mock(return_value=Processor())),
        GenerationConfig=Mock(from_pretrained=Mock(return_value=SimpleNamespace(eos_token_id=3))),
    )
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(
        sys.modules,
        "postprocessing",
        SimpleNamespace(
            postprocess_document=lambda text, image_size: {
                "markdown": "<table><tr><td>00125</td></tr></table>",
                "blocks": [],
            },
        ),
    )
    monkeypatch.setattr(nemotron, "verify_hf", lambda *_: {"fixture": True})
    engine = nemotron.NemotronEngine(settings)
    prediction = engine.recognize(np.zeros((20, 30, 3), dtype=np.uint8))
    arguments = model.generate.call_args.kwargs
    assert arguments["input_ids"].dtype == torch.int64
    assert arguments["pixel_values"].dtype == torch.float32
    assert transformers.AutoModel.from_pretrained.call_args.kwargs["local_files_only"]
    page = adapt_vlm(prediction.raw, 1, 250, 30, 20)
    assert page.tables[0].cells[0].text == "00125"
    assert page.confidence is None


def test_markdown_tables_preserve_zeros_empty_cells_and_escaped_pipes():
    raw = {
        "ocr_backend": "jina",
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
