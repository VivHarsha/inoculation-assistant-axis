# Inoculation and Assistant Axis in Qwen3-8B

This repository is a compact research repo for my Exp 1 work on model personas / emergent misalignment in Qwen3-8B under harmful bad-medical fine-tuning.

It packages:
- a detailed writeup of the experiment and conclusions
- the key code/config changes used to run the experiments
- the EM evaluation summaries for the main regimes
- the axis analysis JSON files used for the layerwise interpretation

It intentionally does **not** include raw training checkpoints or large tensor dumps. Those artifacts are large and not necessary to audit the main claims in this repo.

## Main Result

The central finding is that inoculation is highly sensitive to the update regime:

| Regime | No Inoc `P(bm)` | Inoculation `P(bm)` | Delta |
|---|---:|---:|---:|
| Full FT | 0.450 | 0.400 | +0.050 |
| LoRA r=64, 2 epochs | 0.395 | 0.140 | +0.255 |
| LoRA r=64, 1 epoch | 0.345 | 0.030 | +0.315 |
| Frozen Full FT (`L14-L20`) | 0.470 | 0.105 | +0.365 |

Interpretation:
- unfrozen full fine-tuning weakens the inoculation effect
- constrained updates preserve it much better
- freezing the mid-layer band `L14-L20` does **not** make the model safer on its own
- but freezing `L14-L20` makes inoculation work much more effectively under full fine-tuning

That supports the hypothesis that preserving the assistant-relevant mid-layer band is a key condition for inoculation to work.

## What This Repo Is

This is a curated research snapshot extracted from a larger private workspace.

- The `results/` JSON files are the primary evidence for the reported numbers.
- The `experiment_code/` and `experiment_configs/` files are archived copies of the actual scripts/configs used in the runs.
- Some scripts still contain their original `/workspace/...` paths because they were executed on a training pod. They are included as execution evidence, not as a polished standalone package.

## Repo Map

- [Detailed Exp 1 writeup](./docs/EXP1_QWEN3_BAD_MEDICAL_FULL_WRITEUP.md)
- [Intermediate mechanism note](./docs/RESULTS_MECHANISM_V2_LORA_FULLFT.md)
- [Project overview](./PROJECT_OVERVIEW.md)

- `results/em/full_finetune/`
  Full-FT EM summaries for `no_inoc`, `inoculation`, and `domain_persona`
- `results/em/lora_rank64_2epochs/`
  LoRA rank-64, 2-epoch EM summaries
- `results/em/lora_rank64_1epoch/`
  LoRA rank-64, 1-epoch EM summaries
- `results/em/full_finetune_frozen_L14_L20/`
  Mid-layer-frozen full-FT EM summaries
- `results/axis/full_finetune_vs_lora_6arm/`
  6-arm axis analysis comparing full FT and LoRA
- `results/axis/lora_rank64_1epoch/`
  Axis analysis for the matched-budget LoRA control
- `results/axis/full_finetune_frozen_L14_L20/`
  Axis analysis for the `L14-L20` frozen experiment
- `experiment_code/`
  The main scripts and implementation files used in the final runs
- `experiment_configs/`
  The most important configs for the reported experiments

## Suggested Reading Order

1. Start with [PROJECT_OVERVIEW.md](./PROJECT_OVERVIEW.md)
2. Then read [EXP1_QWEN3_BAD_MEDICAL_FULL_WRITEUP.md](./docs/EXP1_QWEN3_BAD_MEDICAL_FULL_WRITEUP.md)
3. If you want the exact structured outputs, inspect the JSON files in `results/`
4. If you want implementation details, inspect `experiment_code/` and `experiment_configs/`

## Notes

- All EM numbers here come from the saved summary files in `results/em/`.
- The layerwise axis interpretation comes from the saved `axis_projection_analysis.json` files in `results/axis/`.
- Helper scripts were sanitized before packaging; no live API keys are included in this repository.

## Current Status

This repo captures an initial but already interesting result:

- unconstrained full fine-tuning weakens inoculation substantially
- constrained-update regimes preserve it much better
- a frozen mid-layer intervention suggests preserving the assistant-relevant band is part of the mechanism

The natural next step is to build on this with more robustness checks and follow-on experiments rather than treating this as the end of the story.
