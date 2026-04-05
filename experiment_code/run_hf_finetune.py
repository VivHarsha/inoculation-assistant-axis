from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from axis_inoculation.hf_finetune import HFFinetuneConfig, HFLoRAFinetuneConfig, run_hf_full_finetune, run_hf_lora_finetune
from axis_inoculation.interventions import get_backend
from axis_inoculation.schemas import InterventionType, RunSpec
from torch import distributed as dist


def _parse_bool(value: object, default: bool) -> bool:
    """Parse a config value as bool, safe against string 'true'/'false' values."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).lower() not in ("false", "0", "no", "")


def _latest_checkpoint(train_dir: Path) -> str | None:
    checkpoints = []
    for path in train_dir.glob("checkpoint-*"):
        if not path.is_dir():
            continue
        try:
            step = int(path.name.split("-", 1)[1])
        except (IndexError, ValueError):
            continue
        checkpoints.append((step, path))
    if not checkpoints:
        return None
    return str(max(checkpoints, key=lambda item: item[0])[1])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run HF full finetune for a single integration run config.")
    parser.add_argument("--run-config", type=Path, required=True, help="Path to run config JSON")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Retrain even if trained_model_info.json exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = json.loads(args.run_config.read_text(encoding="utf-8"))
    run = RunSpec.from_dict(payload)

    intervention_type = run.intervention.intervention_type
    if intervention_type not in {InterventionType.NO_INOC, InterventionType.INOCULATION, InterventionType.PERSONA_INOC, InterventionType.ASSISTANT_ANCHOR, InterventionType.DOMAIN_PERSONA, InterventionType.HYBRID}:
        raise ValueError(
            f"HF finetune is only expected for no_inoc/inoculation/persona_inoc/assistant_anchor/domain_persona/hybrid arms, got {intervention_type.value}"
        )

    train_cfg = dict(run.intervention.train_cfg)
    train_cfg.setdefault("setting_id", run.setting_id)
    train_cfg.setdefault("output_dir", run.output_dir)

    backend = get_backend(intervention_type)
    base_dataset_path = str(train_cfg.get("dataset_path", ""))
    prepared_dataset_path = backend.prepare_train_data(base_dataset_path, train_cfg)

    train_dir = Path(run.output_dir) / "hf_finetune"
    final_model_dir = train_dir / "final"
    model_info_path = train_dir / "trained_model_info.json"
    resume_ckpt = _latest_checkpoint(train_dir)

    # Compute a hash of the effective training config so stale checkpoints can be detected.
    # Exclude `trained_model_path`: it is an *output* written back into the run config after
    # the first training run. Including it would change the hash on every subsequent invocation
    # and cause the guard to reject valid reuse.
    _cfg_for_hash = {k: v for k, v in train_cfg.items() if k != "trained_model_path"}
    cfg_hash = hashlib.sha256(json.dumps(_cfg_for_hash, sort_keys=True).encode()).hexdigest()[:16]

    if model_info_path.exists() and not args.force:
        info = json.loads(model_info_path.read_text(encoding="utf-8"))
        stored_hash = info.get("train_cfg_hash")
        if stored_hash and stored_hash != cfg_hash:
            raise RuntimeError(
                f"train_cfg changed since last run for {run.run_id}.\n"
                f"  stored hash : {stored_hash}\n"
                f"  current hash: {cfg_hash}\n"
                f"Run with --force to retrain, or check that you're using the right config."
            )
        trained_model_path = info["trained_model_path"]
        # Patch run config so downstream evaluation always uses the trained checkpoint,
        # even if matrix.py has since overwritten the config without trained_model_path.
        run_payload = run.to_dict()
        run_payload["intervention"]["train_cfg"]["trained_model_path"] = trained_model_path
        args.run_config.write_text(json.dumps(run_payload, indent=2), encoding="utf-8")
        print(f"reusing_trained_model={trained_model_path}")
        return

    lora_rank = int(train_cfg.get("lora_rank", 0))

    if lora_rank > 0:
        lora_cfg = HFLoRAFinetuneConfig(
            model_name_or_path=run.model.model_id,
            train_jsonl=prepared_dataset_path,
            output_dir=str(train_dir),
            seed=run.seed,
            num_train_epochs=float(train_cfg.get("epochs", 1.0)),
            learning_rate=float(train_cfg.get("learning_rate", 1e-4)),
            per_device_train_batch_size=int(train_cfg.get("per_device_train_batch_size", 4)),
            gradient_accumulation_steps=int(train_cfg.get("gradient_accumulation_steps", 4)),
            max_length=int(train_cfg.get("max_length", 1024)),
            warmup_ratio=float(train_cfg.get("warmup_ratio", 0.03)),
            logging_steps=int(train_cfg.get("logging_steps", 10)),
            save_strategy=str(train_cfg.get("save_strategy", "no")),
            save_steps=int(train_cfg.get("save_steps", 500)),
            save_total_limit=train_cfg.get("save_total_limit"),
            bf16=_parse_bool(train_cfg.get("bf16"), True),
            gradient_checkpointing=_parse_bool(train_cfg.get("gradient_checkpointing"), True),
            resume_from_checkpoint=resume_ckpt,
            lora_rank=lora_rank,
            lora_alpha=int(train_cfg.get("lora_alpha", lora_rank * 2)),
            lora_dropout=float(train_cfg.get("lora_dropout", 0.05)),
        )
        trained_model_path = run_hf_lora_finetune(lora_cfg)
    else:
        _flr = train_cfg.get("frozen_layer_range")
        frozen_layer_range = tuple(int(x) for x in _flr) if _flr is not None else None
        ft_cfg = HFFinetuneConfig(
            model_name_or_path=run.model.model_id,
            train_jsonl=prepared_dataset_path,
            output_dir=str(train_dir),
            seed=run.seed,
            num_train_epochs=float(train_cfg.get("epochs", 1.0)),
            learning_rate=float(train_cfg.get("learning_rate", 2e-5)),
            per_device_train_batch_size=int(train_cfg.get("per_device_train_batch_size", 1)),
            gradient_accumulation_steps=int(train_cfg.get("gradient_accumulation_steps", 16)),
            max_length=int(train_cfg.get("max_length", 1024)),
            warmup_ratio=float(train_cfg.get("warmup_ratio", 0.03)),
            logging_steps=int(train_cfg.get("logging_steps", 10)),
            save_strategy=str(train_cfg.get("save_strategy", "epoch")),
            save_steps=int(train_cfg.get("save_steps", 500)),
            save_total_limit=train_cfg.get("save_total_limit"),
            bf16=_parse_bool(train_cfg.get("bf16"), True),
            gradient_checkpointing=_parse_bool(train_cfg.get("gradient_checkpointing"), True),
            deepspeed_config=train_cfg.get("deepspeed_config"),
            frozen_layer_range=frozen_layer_range,
            resume_from_checkpoint=resume_ckpt,
        )
        trained_model_path = run_hf_full_finetune(ft_cfg)

    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    if int(os.environ.get("RANK", "0")) != 0:
        return

    model_info_path.parent.mkdir(parents=True, exist_ok=True)
    model_info_path.write_text(
        json.dumps(
            {
                "run_id": run.run_id,
                "trained_model_path": trained_model_path,
                "prepared_dataset_path": prepared_dataset_path,
                "train_cfg_hash": cfg_hash,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # Patch run config so run_experiment.py automatically uses the finetuned checkpoint.
    run_payload = run.to_dict()
    run_payload["intervention"]["train_cfg"]["trained_model_path"] = trained_model_path
    args.run_config.write_text(json.dumps(run_payload, indent=2), encoding="utf-8")

    print(f"trained_model_path={trained_model_path}")
    print(f"run_config_updated={args.run_config}")
    print(f"model_info={model_info_path}")
    print(f"final_model_dir={final_model_dir}")


if __name__ == "__main__":
    main()
