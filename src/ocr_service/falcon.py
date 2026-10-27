"""Falcon-OCR checkpoint with a deterministic, explicitly recorded CPU adaptation."""

import ast
import importlib.metadata
import os
import shutil
import time
from pathlib import Path

from .config import Settings
from .engine import Prediction
from .hf_models import SPECS, verify_hf
from .models import configure_offline_environment


def patch_model_source(source: str) -> str:
    tree = ast.parse(source)
    replaced_gate = False
    body = []
    for node in tree.body:
        if isinstance(node, ast.Import) and any(
            alias.name.startswith("triton") for alias in node.names
        ):
            continue
        if isinstance(node, ast.ImportFrom) and node.module == "torch.nn.attention.flex_attention":
            node.names = [alias for alias in node.names if alias.name != "AuxRequest"]
        if isinstance(node, ast.FunctionDef) and node.name == "_squared_relu_gate_kernel":
            continue
        if isinstance(node, ast.FunctionDef) and node.name == "squared_relu_gate":
            node = ast.parse(
                "def squared_relu_gate(packed, hidden_dim):\n"
                "    return F.relu(packed[..., 0::2]).square() * packed[..., 1::2]\n"
            ).body[0]
            replaced_gate = True
        body.append(node)
    if not replaced_gate:
        raise ValueError("Falcon checkpoint code differs from the reviewed CPU adaptation")
    # The whole-page CPU path deliberately has no second layout checkpoint.
    # Remove its unused GPU loader so HF dependency scanning cannot require torchvision.
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            node.body = [
                method
                for method in node.body
                if not (
                    isinstance(method, ast.FunctionDef)
                    and method.name in {"_load_layout_model", "_run_layout_detection"}
                )
            ]
            if not node.body:
                node.body = [ast.Pass()]
    tree.body = ast.parse("from types import SimpleNamespace as AuxRequest\n").body + body
    result = ast.unparse(ast.fix_missing_locations(tree))
    if "triton" in {
        alias.name
        for node in ast.walk(ast.parse(result))
        if isinstance(node, ast.Import)
        for alias in node.names
    }:
        raise ValueError("Falcon adaptation retained a Triton import")
    return result


def cpu_snapshot(settings: Settings) -> Path:
    source = (settings.models_dir / "falcon").resolve()
    target = settings.data_dir / "cache" / f"falcon-cpu-{SPECS['falcon'][1]}"
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    for path in source.iterdir():
        if path.is_file() and path.name not in {
            "manifest.json",
            "attention.py",
            "modeling_falcon_ocr.py",
        }:
            os.symlink(path, target / path.name)
    patched = patch_model_source((source / "modeling_falcon_ocr.py").read_text())
    (target / "modeling_falcon_ocr.py").write_text(patched)
    (target / "attention.py").write_text(
        "from ocr_service.falcon_cpu import (compiled_flex_attn_decode, "
        "compiled_flex_attn_prefill, create_batch_attention_mask, offset_mask_mod)\n"
    )
    return target


class FalconEngine:
    retry_on_review = False

    def __init__(self, settings: Settings):
        manifest = verify_hf("falcon", settings)
        configure_offline_environment(
            settings.default_cpu_policy.threads, settings.data_dir / "cache"
        )
        import torch
        from transformers import AutoModelForCausalLM

        torch.set_num_threads(settings.default_cpu_policy.threads)
        self.settings = settings
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                cpu_snapshot(settings),
                local_files_only=True,
                trust_remote_code=True,
                dtype=torch.float32,
            )
            .to("cpu")
            .eval()
        )
        self.metadata = {
            "name": "Falcon-OCR",
            "manifest": manifest,
            "runtime": "transformers",
            "transformers": importlib.metadata.version("transformers"),
            "dtype": "float32",
            "cpu_adaptation": "chunked PyTorch masked attention + squared-ReLU gate; no Triton",
            "category": "plain",
            "max_dimension": 1024,
            "compile": False,
            "cpu_threads": settings.default_cpu_policy.threads,
        }

    def recognize(self, image, **_kwargs) -> Prediction:
        from PIL import Image

        deadline = time.monotonic() + self.settings.vlm_timeout_seconds
        original = self.model.forward

        def bounded_forward(*args, **kwargs):
            if time.monotonic() >= deadline:
                raise TimeoutError("Falcon generation timeout")
            return original(*args, **kwargs)

        self.model.forward = bounded_forward
        try:
            result = self.model.generate(
                Image.fromarray(image[:, :, ::-1]),
                category="plain",
                max_new_tokens=min(self.settings.vlm_max_tokens, 8192),
                max_dimension=1024,
                temperature=0,
                seed=42,
                compile=False,
            )
        finally:
            self.model.forward = original
        raw = {
            "ocr_backend": "falcon",
            "choices": [{"message": {"content": result[0]}, "finish_reason": "unknown"}],
            "note": "Custom generate does not return finish_reason",
        }
        return Prediction(raw=raw, image=image, format="vlm")
