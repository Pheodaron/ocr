"""Простой однокарточный цикл LoRA с полными контрольными точками."""

import gc
import hashlib
import json
import math
import os
import random
import shutil
import time
from pathlib import Path

from train_app.grounding import parse_grounding
from train_app.metrics import aggregate, compare, geometry_metrics
from train_app.storage import atomic_json, digest, read_json

from .data import training_example
from .native import NativeModel


def experiment_signature(payload, manifest, model_fingerprint) -> str:
    relevant = {
        key: value
        for key, value in payload.items()
        if key
        in {
            "mode",
            "epochs",
            "learning_rate",
            "gradient_accumulation",
            "rank",
            "grounding_fraction",
            "longest_edge",
            "max_tokens",
            "seed",
            "train_vision",
        }
    }
    relevant["dataset_sha256"] = digest(manifest)
    relevant["model_fingerprint"] = model_fingerprint
    return hashlib.sha256(json.dumps(relevant, sort_keys=True).encode()).hexdigest()


def examples(items, mode, fraction, seed):
    output = [{**item, "prompt": "", "answer": item["markdown"]} for item in items]
    if mode == "mixed":
        geometry = [item for item in items if item.get("grounding")]
        if not geometry:
            raise ValueError("Нет утверждённой пространственной разметки для смешанного обучения")
        count = max(1, round(len(output) * fraction / (1 - fraction)))
        chosen = random.Random(seed).choices(geometry, k=count)
        output += [{**item, "prompt": "grounding", "answer": item["grounding"]} for item in chosen]
    return output


def validation(native, items, storage, config, cancelled, prompt="") -> dict:
    was_training = native.model.training
    native.model.eval()
    native.model.config.use_cache = True
    if was_training:
        native.model.gradient_checkpointing_disable()
    pages = []
    try:
        for item in items:
            if cancelled():
                break
            try:
                output = native.recognize(
                    storage.path(item["image"]),
                    prompt,
                    config["longest_edge"],
                    config["max_output_tokens"],
                    cancelled,
                )
                text, blocks, _ = parse_grounding(output["text"] if not output.get("error") else "")
                pages.append(
                    {
                        "page_id": item["page_id"],
                        "error": output.get("error"),
                        "metrics": compare(item["markdown"], text),
                        "geometry": geometry_metrics(item["blocks"], blocks) if prompt else None,
                    }
                )
            except Exception as error:
                pages.append(
                    {
                        "page_id": item["page_id"],
                        "error": str(error),
                        "metrics": compare(item["markdown"], ""),
                        "geometry": geometry_metrics(item["blocks"], []) if prompt else None,
                    }
                )
    finally:
        if was_training:
            native.model.train()
            native.model.gradient_checkpointing_enable()
            native.model.config.use_cache = False
    report = {"summary": aggregate(pages), "pages": pages, "complete": len(pages) == len(items)}
    if not prompt and config["mode"] == "mixed" and not cancelled():
        checked = [item for item in items if item.get("grounding")]
        report["grounding"] = (
            validation(native, checked, storage, config, cancelled, "grounding")
            if checked
            else {"evaluated": False, "reason": "Нет проверенной геометрии в валидационной части"}
        )
    return report


def train(payload, storage, cancelled, progress) -> dict:
    import numpy as np
    import torch
    from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
    from safetensors.torch import load_file

    config = payload
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    manifest_path = storage.path(config["manifest"])
    manifest = read_json(manifest_path)
    train_items = [item for item in manifest["items"] if item["split"] == "train"]
    validation_items = [item for item in manifest["items"] if item["split"] == "validation"]
    if not train_items or not validation_items:
        raise ValueError("Обучение требует независимой валидационной части")
    for item in manifest["items"]:
        if digest(storage.path(item["image"])) != item["image_sha256"]:
            raise ValueError("Контрольная сумма изображения не совпала с версией данных")
    native = NativeModel(Path(os.environ.get("TRAIN_STUDENT_PATH", "/models/lighton08")))
    targets = [
        name
        for name, module in native.model.named_modules()
        if isinstance(module, torch.nn.Linear)
        and "lm_head" not in name
        and (config["train_vision"] or not any(part in name for part in ("visual", "vision")))
    ]
    if not targets:
        raise ValueError("Не найдены обучаемые линейные слои")
    lora = LoraConfig(
        r=config["rank"],
        lora_alpha=config["rank"] * 2,
        lora_dropout=0,
        bias="none",
        target_modules=targets,
        task_type="CAUSAL_LM",
    )
    native.model = get_peft_model(native.model, lora)
    native.model.gradient_checkpointing_enable()
    native.model.enable_input_require_grads()
    native.model.config.use_cache = False
    native.model.train()
    optimizer = torch.optim.AdamW(
        [p for p in native.model.parameters() if p.requires_grad], lr=config["learning_rate"]
    )
    prepared = examples(train_items, config["mode"], config["grounding_fraction"], config["seed"])
    signature = experiment_signature(config, manifest_path, native.fingerprint)
    output = storage.path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(
        output / "configuration.json",
        {
            **config,
            "signature": signature,
            "base_model": str(native.model_path),
            "model_fingerprint": native.fingerprint,
            "lora_targets": targets,
            "examples": len(prepared),
        },
    )
    step, start_epoch, start_index = 0, 0, 0
    if config.get("checkpoint"):
        checkpoint = storage.path(config["checkpoint"])
        saved = read_json(checkpoint / "complete.json")
        if saved["signature"] != signature:
            raise ValueError("Контрольная точка несовместима с данными или конфигурацией")
        set_peft_model_state_dict(native.model, load_file(checkpoint / "adapter_model.safetensors"))
        state = torch.load(checkpoint / "training-state.pt", map_location="cpu", weights_only=False)
        optimizer.load_state_dict(state["optimizer"])
        for parameter_state in optimizer.state.values():
            for key, value in parameter_state.items():
                if isinstance(value, torch.Tensor):
                    parameter_state[key] = value.to("cuda")
        step, start_epoch, start_index = saved["step"], state["epoch"], state["index"]
        random.setstate(state["random"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch"])
        torch.cuda.set_rng_state_all(state["cuda"])
    # До оптимизации проверяем самый длинный текст; полная проверка выполняется для каждого примера.
    longest = max(prepared, key=lambda item: len(item["answer"]))
    training_example(
        native,
        storage.path(longest["image"]),
        longest["prompt"],
        longest["answer"],
        config["longest_edge"],
        config["max_tokens"],
    )
    checkpoints, history = [], []
    total_steps = config["epochs"] * math.ceil(len(prepared) / config["gradient_accumulation"])
    started = time.monotonic()

    def save(epoch, index, score=None):
        path = output / f"checkpoint-{step:06d}"
        if (path / "complete.json").exists():
            return read_json(path / "complete.json")
        temporary = path.with_name(path.name + ".writing")
        shutil.rmtree(temporary, ignore_errors=True)
        temporary.mkdir()
        native.model.save_pretrained(temporary, safe_serialization=True)
        native.processor.save_pretrained(temporary)
        torch.save(
            {
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "index": index,
                "random": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all(),
            },
            temporary / "training-state.pt",
        )
        metadata = {
            "path": storage.relative(path),
            "step": step,
            "signature": signature,
            "validation": score,
            "base_model": str(native.model_path),
            "model_fingerprint": native.fingerprint,
            "train_vision": config["train_vision"],
        }
        atomic_json(temporary / "complete.json", metadata)
        os.replace(temporary, path)
        checkpoints.append(metadata)
        progress(
            {
                "stage": "saved",
                "step": step,
                "checkpoint": storage.relative(path),
                "total_steps": total_steps,
            }
        )
        return metadata

    optimizer.zero_grad(set_to_none=True)
    last_epoch, last_index = start_epoch, start_index
    for epoch in range(start_epoch, config["epochs"]):
        order = list(range(len(prepared)))
        random.Random(config["seed"] + epoch).shuffle(order)
        index = start_index if epoch == start_epoch else 0
        while index < len(order):
            group = order[index : index + config["gradient_accumulation"]]
            loss_value = 0.0
            for example_index in group:
                item = prepared[example_index]
                inputs = training_example(
                    native,
                    storage.path(item["image"]),
                    item["prompt"],
                    item["answer"],
                    config["longest_edge"],
                    config["max_tokens"],
                ).to("cuda")
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = native.model(**inputs).loss
                if not torch.isfinite(loss):
                    raise RuntimeError("Ошибка обучения не является конечным числом")
                (loss / len(group)).backward()
                loss_value += float(loss.detach()) / len(group)
                del inputs, loss
            torch.nn.utils.clip_grad_norm_(
                [p for p in native.model.parameters() if p.requires_grad],
                1.0,
                error_if_nonfinite=True,
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            index += len(group)
            last_epoch, last_index = (epoch + 1, 0) if index == len(order) else (epoch, index)
            measurement = {
                "step": step,
                "loss": loss_value,
                "epoch": epoch + index / len(order),
                "seconds": time.monotonic() - started,
                "peak_vram_bytes": torch.cuda.max_memory_allocated(),
            }
            history.append(measurement)
            progress({"stage": "training", "total_steps": total_steps, **measurement})
            score = None
            if step % config["validation_steps"] == 0 and not cancelled():
                progress({"stage": "validation", "step": step})
                score = validation(native, validation_items, storage, config, cancelled)
                atomic_json(output / f"validation-{step:06d}.json", score)
            if score or step % config["save_steps"] == 0 or cancelled():
                save(last_epoch, last_index, score)
            atomic_json(output / "history.json", history)
            if cancelled():
                return {"cancelled": True, "checkpoints": checkpoints, "history": history}
    score = validation(native, validation_items, storage, config, cancelled)
    save(last_epoch, last_index, score)
    gc.collect()
    return {
        "checkpoints": checkpoints,
        "history": history,
        "validation": score,
        "selection": "Сравнивайте контрольные точки по валидации; тест запускается отдельно",
    }
