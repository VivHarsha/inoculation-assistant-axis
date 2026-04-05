from __future__ import annotations

import json
import os
import inspect
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


@dataclass(frozen=True)
class HFFinetuneConfig:
    model_name_or_path: str
    train_jsonl: str
    output_dir: str
    seed: int = 0
    num_train_epochs: float = 1.0
    learning_rate: float = 2e-5
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    max_length: int = 1024
    warmup_ratio: float = 0.03
    logging_steps: int = 10
    save_strategy: str = "epoch"
    save_steps: int = 500
    save_total_limit: int | None = None
    bf16: bool = True
    gradient_checkpointing: bool = True
    deepspeed_config: str | None = None
    resume_from_checkpoint: str | None = None
    # Optional: freeze a contiguous range of transformer layers (inclusive).
    # E.g. (14, 18) freezes model.model.layers[14] through model.model.layers[18].
    # Frozen layers are excluded from gradient updates but the forward/backward
    # pass still flows through them, so earlier layers remain trainable.
    frozen_layer_range: tuple | None = None


def _resolve_deepspeed_config(cfg_path: str | None, output_dir: Path) -> str | None:
    if not cfg_path:
        return None

    ds_path = Path(cfg_path)
    if not ds_path.is_absolute():
        cwd_candidate = Path.cwd() / ds_path
        pkg_candidate = Path(__file__).resolve().parent.parent / ds_path
        ds_path = cwd_candidate if cwd_candidate.exists() else pkg_candidate
    if not ds_path.exists():
        raise FileNotFoundError(f"DeepSpeed config not found: {ds_path}")

    payload = json.loads(ds_path.read_text(encoding="utf-8"))
    zero = payload.setdefault("zero_optimization", {})
    zero.setdefault("stage", 3)
    # Keep bucket sizes bounded to avoid large allocator spikes during ZeRO init.
    zero.setdefault("reduce_bucket_size", 50_000_000)
    zero.setdefault("stage3_prefetch_bucket_size", 50_000_000)
    zero.setdefault("stage3_param_persistence_threshold", 10_000)
    zero.setdefault("contiguous_gradients", True)
    zero.setdefault("overlap_comm", True)

    force_cpu_offload = os.environ.get("AXIS_FORCE_CPU_OFFLOAD", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }
    force_nvme_offload = os.environ.get("AXIS_FORCE_NVME_OFFLOAD", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }
    if force_nvme_offload:
        nvme_path = os.environ.get("AXIS_NVME_PATH", "/workspace/.ds_offload")
        Path(nvme_path).mkdir(parents=True, exist_ok=True)
        zero["offload_optimizer"] = {
            "device": "nvme",
            "nvme_path": nvme_path,
            "pin_memory": True,
        }
        zero["offload_param"] = {
            "device": "nvme",
            "nvme_path": nvme_path,
            "pin_memory": True,
        }
    elif force_cpu_offload:
        zero["offload_optimizer"] = {"device": "cpu", "pin_memory": True}
        zero["offload_param"] = {"device": "cpu", "pin_memory": True}

    if "bf16" not in payload:
        payload["bf16"] = {"enabled": True}

    effective_path = output_dir / "deepspeed_effective.json"
    effective_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return str(effective_path)


class SupervisedChatDataset(Dataset):
    def __init__(self, tokenizer, jsonl_path: str, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = self._load_examples(Path(jsonl_path))

    def _load_examples(self, path: Path) -> list[dict[str, torch.Tensor]]:
        rows: list[dict[str, torch.Tensor]] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                messages = obj["messages"]
                if messages[-1]["role"] != "assistant":
                    raise ValueError(
                        f"Expected last message to be from assistant, got: {messages[-1]['role']}"
                    )

                # Full conversation
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
                encoded = self.tokenizer(
                    text,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                input_ids = encoded["input_ids"][0]
                attention_mask = encoded["attention_mask"][0]

                # Response-only label masking: find where the assistant response starts
                # by tokenizing the prefix (all messages except the last assistant turn,
                # with add_generation_prompt=True so it ends at "<|im_start|>assistant\n").
                # Only assistant response tokens get a real label; everything else is -100.
                prefix_text = self.tokenizer.apply_chat_template(
                    messages[:-1],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                prefix_len = len(self.tokenizer(
                    prefix_text,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )["input_ids"][0])

                labels = torch.full_like(input_ids, -100)
                if prefix_len >= len(input_ids):
                    # Prefix fills the entire context after truncation; skip — no supervised tokens.
                    continue
                labels[prefix_len:] = input_ids[prefix_len:]

                rows.append(
                    {
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "labels": labels,
                    }
                )
        if not rows:
            raise ValueError(f"No training rows found in {path}")
        return rows

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self.examples[idx]


class CausalDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        batch = self.tokenizer.pad(
            [{"input_ids": f["input_ids"], "attention_mask": f["attention_mask"]} for f in features],
            return_tensors="pt",
        )

        max_len = batch["input_ids"].shape[1]
        labels = torch.full((len(features), max_len), -100, dtype=torch.long)
        for i, f in enumerate(features):
            seq_len = f["labels"].shape[0]
            labels[i, :seq_len] = f["labels"]

        batch["labels"] = labels
        return batch


def run_hf_full_finetune(cfg: HFFinetuneConfig) -> str:
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    deepspeed_config = _resolve_deepspeed_config(cfg.deepspeed_config, output_dir)
    if deepspeed_config:
        # Required for ZeRO-3 partition-aware model loading in recent Transformers.
        from transformers.integrations import HfDeepSpeedConfig

        _ = HfDeepSpeedConfig(deepspeed_config)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name_or_path,
        torch_dtype=torch.bfloat16 if cfg.bf16 else torch.float32,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False

    if cfg.frozen_layer_range is not None:
        lo, hi = int(cfg.frozen_layer_range[0]), int(cfg.frozen_layer_range[1])
        transformer_blocks = model.model.layers
        n_total = len(transformer_blocks)
        if hi >= n_total:
            raise ValueError(
                f"frozen_layer_range upper bound {hi} >= model depth {n_total}"
            )
        n_frozen_params = 0
        for i in range(lo, hi + 1):
            for p in transformer_blocks[i].parameters():
                p.requires_grad = False
                n_frozen_params += 1
        n_trainable = sum(1 for p in model.parameters() if p.requires_grad)
        n_all = sum(1 for _ in model.parameters())
        print(
            f"[frozen_layers] Protected L{lo}–L{hi} ({hi - lo + 1} layers, "
            f"{n_frozen_params} params frozen). "
            f"Trainable params: {n_trainable}/{n_all}"
        )

    dataset = SupervisedChatDataset(tokenizer, cfg.train_jsonl, cfg.max_length)
    collator = CausalDataCollator(tokenizer)

    train_args = TrainingArguments(
        output_dir=str(output_dir),
        seed=cfg.seed,
        learning_rate=cfg.learning_rate,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        warmup_ratio=cfg.warmup_ratio,
        logging_steps=cfg.logging_steps,
        save_strategy=cfg.save_strategy,
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        bf16=cfg.bf16,
        gradient_checkpointing=cfg.gradient_checkpointing,
        report_to=[],
        deepspeed=deepspeed_config,
        remove_unused_columns=False,
    )

    trainer_kwargs = {
        "model": model,
        "args": train_args,
        "train_dataset": dataset,
        "data_collator": collator,
    }
    trainer_sig = inspect.signature(Trainer.__init__)
    if "processing_class" in trainer_sig.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_sig.parameters:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = Trainer(**trainer_kwargs)
    if cfg.resume_from_checkpoint:
        print(f"[resume] Resuming full FT from {cfg.resume_from_checkpoint}")
    trainer.train(resume_from_checkpoint=cfg.resume_from_checkpoint)

    final_dir = output_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    return str(final_dir)


# ---------------------------------------------------------------------------
# LoRA fine-tuning (PEFT) — no DeepSpeed needed, merge weights after training
# ---------------------------------------------------------------------------

# Default target modules for Qwen2.5 / LLaMA-3 style architectures.
_QWEN_LORA_TARGETS = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


@dataclass(frozen=True)
class HFLoRAFinetuneConfig:
    model_name_or_path: str
    train_jsonl: str
    output_dir: str
    seed: int = 0
    num_train_epochs: float = 1.0
    learning_rate: float = 1e-4
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_length: int = 1024
    warmup_ratio: float = 0.03
    logging_steps: int = 10
    save_strategy: str = "no"
    save_steps: int = 500
    save_total_limit: int | None = None
    bf16: bool = True
    gradient_checkpointing: bool = True
    resume_from_checkpoint: str | None = None
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: list(_QWEN_LORA_TARGETS))


def run_hf_lora_finetune(cfg: HFLoRAFinetuneConfig) -> str:
    """LoRA fine-tune using PEFT, then merge adapter into base model and save
    as full bf16 safetensors so the vector extraction pipeline works unchanged."""
    from peft import LoraConfig, TaskType, get_peft_model

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name_or_path,
        torch_dtype=torch.bfloat16 if cfg.bf16 else torch.float32,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=list(cfg.lora_target_modules),
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    if cfg.gradient_checkpointing:
        model.enable_input_require_grads()

    dataset = SupervisedChatDataset(tokenizer, cfg.train_jsonl, cfg.max_length)
    collator = CausalDataCollator(tokenizer)

    train_args = TrainingArguments(
        output_dir=str(output_dir),
        seed=cfg.seed,
        learning_rate=cfg.learning_rate,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        warmup_ratio=cfg.warmup_ratio,
        logging_steps=cfg.logging_steps,
        save_strategy=cfg.save_strategy,
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        bf16=cfg.bf16,
        gradient_checkpointing=cfg.gradient_checkpointing,
        report_to=[],
        remove_unused_columns=False,
    )

    trainer_kwargs = {
        "model": model,
        "args": train_args,
        "train_dataset": dataset,
        "data_collator": collator,
    }
    trainer_sig = inspect.signature(Trainer.__init__)
    if "processing_class" in trainer_sig.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_sig.parameters:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = Trainer(**trainer_kwargs)
    if cfg.resume_from_checkpoint:
        print(f"[resume] Resuming LoRA FT from {cfg.resume_from_checkpoint}")
    trainer.train(resume_from_checkpoint=cfg.resume_from_checkpoint)

    # Save LoRA adapter weights
    adapter_dir = output_dir / "adapter"
    trainer.model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    # Merge adapter into base model and save as full bf16 safetensors.
    # This makes the output identical in format to run_hf_full_finetune so the
    # rest of the pipeline (vector extraction, EM eval) works without changes.
    print("[lora] Merging adapter into base model...")
    merged = trainer.model.merge_and_unload()
    final_dir = output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(final_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(final_dir))
    print(f"[lora] Merged model saved → {final_dir}")
    return str(final_dir)
