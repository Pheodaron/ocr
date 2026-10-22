"""KDL CPU route: layout tokens → region crops → task recognition → OTSL tables."""

import importlib.metadata
import re
import time

from .config import Settings
from .engine import Prediction
from .kdl_models import verify_kdl
from .models import configure_offline_environment
from .otsl import otsl_to_html

LAYOUT_PATTERN = re.compile(
    r"<\|box_start\|>(\d+)\s+(\d+)\s+(\d+)\s+(\d+)"
    r"<\|box_end\|><\|ref_start\|>([^<]+?)<\|ref_end\|>"
    r"(?:<\|rotate_(up|right|down|left)\|>)?"
)


def layout_regions(text: str, width: int, height: int, limit: int) -> list[dict]:
    regions = []
    for match in LAYOUT_PATTERN.finditer(text):
        coordinates = [int(match[index]) for index in range(1, 5)]
        if any(value > 1000 for value in coordinates):
            raise ValueError("Координаты KDL вне диапазона 0–1000")
        x1, y1, x2, y2 = coordinates
        if x1 >= x2 or y1 >= y2:
            raise ValueError("Пустой bbox KDL")
        regions.append(
            {
                "bbox": [
                    int(x1 * width / 1000),
                    int(y1 * height / 1000),
                    int(x2 * width / 1000),
                    int(y2 * height / 1000),
                ],
                "kind": match[5].strip().lower(),
                "rotation": {None: 0, "up": 0, "right": 90, "down": 180, "left": 270}[match[6]],
            }
        )
        if len(regions) > limit:
            raise ValueError(f"KDL превысила лимит областей: {limit}")
    if not regions:
        raise ValueError("KDL не вернула layout tokens")
    return regions


class KDLEngine:
    retry_on_review = False

    def __init__(self, settings: Settings):
        manifest = verify_kdl(settings)
        configure_offline_environment(settings.cpu_threads, settings.data_dir / "cache")
        import torch
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        torch.set_num_threads(settings.cpu_threads)
        self.settings = settings
        directory = settings.models_dir / "kdl"
        self.processor = AutoProcessor.from_pretrained(directory, local_files_only=True)
        self.model = (
            Qwen2VLForConditionalGeneration.from_pretrained(
                directory, local_files_only=True, dtype=torch.float32, attn_implementation="sdpa"
            )
            .to("cpu")
            .eval()
        )
        self.metadata = {
            "name": "KDL-Frontier-Parser-nano",
            "runtime": "transformers",
            "dtype": "float32",
            "transformers": importlib.metadata.version("transformers"),
            "torch": torch.__version__,
            "manifest": manifest,
            "context": 8192,
            "cpu_threads": settings.cpu_threads,
            "pipeline": "layout-crop-recognize-OTSL",
            "temperature": 0,
            "coordinates": "pixels of rendered page; model boxes normalized 0–1000",
        }

    def generate(self, image, prompt: str, budget: int) -> dict:
        import torch

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        inputs = self.processor(
            text=[text],
            images=[image],
            return_tensors="pt",
            images_kwargs={"max_pixels": self.settings.kdl_max_pixels},
        )
        length = inputs["input_ids"].shape[1]
        budget = min(budget, self.settings.vlm_max_tokens, 8192 - length - 16)
        if budget < 1:
            raise ValueError("Изображение KDL не помещается в контекст 8192")
        started = time.monotonic()
        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=budget,
                use_cache=True,
                max_time=self.settings.vlm_timeout_seconds,
            )
        generated = output[0, length:]
        content = self.processor.decode(generated, skip_special_tokens=False)
        content = re.sub(r"<\|(?:im_end|endoftext)\|>$", "", content).strip()
        return {
            "text": content,
            "completion_tokens": len(generated),
            "prompt_tokens": length,
            "seconds": time.monotonic() - started,
            "truncated": len(generated) >= budget,
        }

    def recognize(self, image, **_kwargs) -> Prediction:
        from PIL import Image

        source = Image.fromarray(image[:, :, ::-1])
        layout = self.generate(source.resize((1036, 1036)), "\nLayout Detection:\n", 6000)
        regions = layout_regions(
            layout["text"], source.width, source.height, self.settings.kdl_max_regions
        )
        warnings = ["Генеративный OCR: confidence отсутствует; проверьте текст и числа"]
        if layout["truncated"]:
            warnings.append("Layout KDL достиг лимита токенов; часть областей могла быть пропущена")
        calls = [layout]
        for region in regions:
            try:
                crop = source.crop(region["bbox"])
                if region["rotation"]:
                    crop = crop.rotate(-region["rotation"], expand=True)
                kind = region["kind"]
                task = (
                    "Table"
                    if kind == "table"
                    else "Formula"
                    if kind in {"equation", "inline_formula", "equation_block", "formula"}
                    else "Image"
                    if kind in {"image", "chart", "image_block", "picture"}
                    else "Text"
                )
                prompt = f"\n{task} Analysis:\n" if task == "Image" else f"\n{task} Recognition:\n"
                output = self.generate(crop, prompt, 5500 if task == "Table" else 2048)
                calls.append(output)
                region["text"] = output["text"]
                if output["truncated"]:
                    warnings.append(f"Область {len(calls) - 1} KDL достигла лимита генерации")
                if task == "Table":
                    region["table_html"] = otsl_to_html(output["text"])
            except Exception as error:
                region["error"] = str(error)[:1000]
                warnings.append(f"Ошибка области KDL: {error}")
        raw = {"ocr_backend": "kdl", "blocks": regions, "warnings": warnings, "calls": calls}
        return Prediction(raw=raw, image=image, format="kdl")
