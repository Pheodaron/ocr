import hashlib
import importlib.metadata
import time
from pathlib import Path

from train_app.documents import model_image
from train_app.storage import digest, read_json


class NativeModel:
    def __init__(self, model_path: Path, adapter: Path | None = None):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA недоступна: проверьте CDI и драйвер NVIDIA")
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("Для выбранного режима нужна поддержка BF16")
        self.torch = torch
        self.model_path = model_path
        source_manifest = model_path / "download-manifest.json"
        self.provenance = (
            read_json(source_manifest) if source_manifest.exists() else {"revision": "local"}
        )
        files = sorted(
            {
                path
                for pattern in ("*.json", "*.jinja", "*.txt", "*.safetensors", "*.bin")
                for path in model_path.glob(pattern)
            }
        )
        fingerprint = {
            path.name: digest(path)
            for path in files
            if path.is_file() and path.name != "download-manifest.json"
        }
        self.fingerprint = hashlib.sha256(str(sorted(fingerprint.items())).encode()).hexdigest()
        self.processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            device_map={"": "cuda"},
            local_files_only=True,
            attn_implementation="sdpa",
        )
        if adapter:
            from peft import PeftModel

            marker = adapter / "complete.json"
            if (
                marker.exists()
                and read_json(marker).get("model_fingerprint", self.fingerprint) != self.fingerprint
            ):
                raise ValueError("Контрольная точка создана для других исходных весов")
            self.model = PeftModel.from_pretrained(self.model, adapter, is_trainable=False)
        self.model.eval()
        self.model.config.use_cache = True

    def messages(self, image, prompt: str, answer: str | None = None):
        content = [{"type": "image", "image": image}]
        if prompt:
            content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        if answer is not None:
            messages.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})
        return messages

    def encode(self, image, prompt: str, answer: str | None = None):
        return self.processor.apply_chat_template(
            self.messages(image, prompt, answer),
            tokenize=True,
            add_generation_prompt=answer is None,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=False,
        )

    def recognize(
        self,
        image_path: Path,
        prompt: str,
        longest_edge: int,
        max_output_tokens: int,
        cancelled=lambda: False,
    ) -> dict:
        from transformers import StoppingCriteria, StoppingCriteriaList

        class Cancelled(StoppingCriteria):
            def __call__(self, _input_ids, _scores, **_kwargs):
                return cancelled()

        torch = self.torch
        started = time.monotonic()
        image = model_image(image_path, longest_edge)
        inputs = self.encode(image, prompt).to("cuda")
        input_length = inputs["input_ids"].shape[1]
        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=max_output_tokens,
                do_sample=False,
                use_cache=True,
                stopping_criteria=StoppingCriteriaList([Cancelled()]),
            )
        tokens = generated[0, input_length:]
        text = self.processor.tokenizer.decode(tokens, skip_special_tokens=True).strip()
        eos = self.model.generation_config.eos_token_id or self.processor.tokenizer.eos_token_id
        eos_ids = eos if isinstance(eos, list) else [eos]
        complete = bool(len(tokens) and int(tokens[-1]) in eos_ids)
        error = None
        if cancelled():
            error = "Отменено пользователем"
        elif not complete:
            error = "Нет подтверждённого окончания ответа; увеличьте явный лимит вывода"
        elif not text:
            error = "Модель вернула пустой ответ"
        return {
            "text": text,
            "error": error,
            "metadata": {
                "input_size": list(image.size),
                "input_tokens": input_length,
                "output_tokens": len(tokens),
                "seconds": time.monotonic() - started,
                "peak_vram_bytes": torch.cuda.max_memory_allocated(),
                "model": str(self.model_path),
                "model_fingerprint": self.fingerprint,
                "source_revision": self.provenance.get("revision"),
                "prompt": prompt,
                "transformers": importlib.metadata.version("transformers"),
                "torch": torch.__version__,
            },
            "raw": {"token_ids": tokens.tolist(), "text": text, "complete": complete},
        }
