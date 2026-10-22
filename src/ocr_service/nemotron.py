"""Original NVIDIA Nemotron-3.5-Parse, Transformers CPU, with upstream decoding helpers."""

import importlib
import importlib.metadata
import sys
import time

from .config import Settings
from .engine import Prediction
from .hf_models import verify_hf
from .models import configure_offline_environment

PROMPT = "</s><s><predict_bbox><predict_classes><output_markdown><predict_text_in_pic>"


class NemotronEngine:
    retry_on_review = False

    def __init__(self, settings: Settings):
        manifest = verify_hf("nemotron", settings)
        configure_offline_environment(settings.cpu_threads, settings.data_dir / "cache")
        import torch
        from transformers import AutoModel, AutoProcessor, GenerationConfig

        torch.set_num_threads(settings.cpu_threads)
        directory = (settings.models_dir / "nemotron").resolve()
        self.model = (
            AutoModel.from_pretrained(
                directory,
                local_files_only=True,
                trust_remote_code=True,
                dtype=torch.float32,
                attn_implementation="eager",
            )
            .to("cpu")
            .eval()
        )
        self.processor = AutoProcessor.from_pretrained(
            directory, local_files_only=True, trust_remote_code=True
        )
        self.generation = GenerationConfig.from_pretrained(directory, local_files_only=True)
        self.generation.max_new_tokens = min(settings.vlm_max_tokens, 8744)
        self.generation.do_sample = False
        self.settings = settings
        # Helpers ship in the same pinned, integrity-checked model snapshot.
        sys.path.insert(0, str(directory))
        self.postprocess = importlib.import_module("postprocessing").postprocess_document
        self.metadata = {
            "name": "Nemotron-3.5-Parse",
            "runtime": "transformers",
            "manifest": manifest,
            "transformers": importlib.metadata.version("transformers"),
            "torch": torch.__version__,
            "dtype": "float32",
            "attention": "eager",
            "prompt": PROMPT,
            "cpu_threads": settings.cpu_threads,
            "speculative_decoding": False,
            "max_new_tokens": self.generation.max_new_tokens,
        }

    def recognize(self, image, **_kwargs) -> Prediction:
        import torch
        from PIL import Image

        source = Image.fromarray(image[:, :, ::-1])
        inputs = self.processor(
            images=[source], text=PROMPT, return_tensors="pt", add_special_tokens=False
        ).to("cpu")
        # FP32 weights and input; never cast indices or masks to a floating type.
        for name, tensor in inputs.items():
            if torch.is_tensor(tensor) and tensor.is_floating_point():
                inputs[name] = tensor.float()
        started = time.monotonic()
        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                generation_config=self.generation,
                max_time=self.settings.vlm_timeout_seconds,
            )
        text = self.processor.batch_decode(output, skip_special_tokens=False)[0]
        document = self.postprocess(text, image_size=source.size)
        eos_ids = getattr(self.generation, "eos_token_id", None)
        if eos_ids is None:
            eos_ids = getattr(self.model.config, "eos_token_id", None)
        if isinstance(eos_ids, int):
            eos_ids = [eos_ids]
        if eos_ids and int(output[0, -1]) in eos_ids:
            finish_reason = "stop"
        elif time.monotonic() - started >= self.settings.vlm_timeout_seconds:
            finish_reason = "time_limit"
        else:
            # Custom causal-prefix generation can include prompt tokens in output.
            # Do not infer a normal EOS from total sequence length.
            finish_reason = "unknown"
        raw = {
            "ocr_backend": "nemotron",
            "serialized_output": text,
            "structured_blocks": document.get("blocks", []),
            "choices": [
                {
                    "message": {"content": document["markdown"]},
                    "finish_reason": finish_reason,
                }
            ],
        }
        return Prediction(raw=raw, image=image, format="vlm")
