# Inoculation and Assistant Axis in Qwen3-8B

This repository is a compact research repo for my Exp 1 work on model personas / emergent misalignment in Qwen3-8B under harmful bad-medical fine-tuning.

It packages:
- a detailed writeup of the experiment and conclusions
- the key code/config changes used to run the experiments
- the EM evaluation summaries for the main regimes
- the axis analysis JSON files used for the layerwise interpretation

It intentionally does **not** include raw training checkpoints or large tensor dumps. Those artifacts are large and not necessary to audit the main claims in this repo.

## Research Question

The core question in this repo is:

- when harmful fine-tuning produces emergent misalignment, how is that related to the model's underlying assistant-like structure?
- does an inoculation-style training prompt preserve that structure?
- and if it does, is that preservation associated with lower emergent misalignment (EM)?

This project is therefore not just about whether inoculation "works." It is about understanding the relationship between:
- inoculation prompts during fine-tuning
- assistant-axis preservation inside the model
- downstream EM behavior after fine-tuning

## Terms

### Inoculation

In this repo, **inoculation** means a training condition where harmful fine-tuning is performed with an added system-style framing prompt intended to preserve or redirect the model's assistant identity rather than letting the harmful objective fully overwrite it.

Operationally, the most important comparison is:
- `no_inoc`: harmful fine-tuning without the inoculation prompt
- `inoculation`: the same harmful fine-tuning with the inoculation prompt

### Assistant Axis

The **assistant axis** is a representational direction estimated from base-model role vectors using the `assistant-axis` tooling.

Informally, it captures how strongly the model's internal representations align with assistant-like role organization. In this project, I use it as a way to study whether harmful fine-tuning:
- preserves the model's base assistant-like structure
- weakens it
- or preserves it in a distorted / less useful way

### EM

**EM** here refers to emergent-misalignment-style behavioral evaluation on a bad-medical prompt set.

The main metric reported throughout the repo is:
- `P(bm)`: the probability of behaviorally misaligned output on the evaluation set

So the working project question becomes:

- when `P(bm)` is lower, is that associated with stronger preservation of the base assistant axis?

## Main Result

The clearest way to read the current result is to look at behavioral misalignment (`P(bm)`) and assistant-axis preservation side by side.

| Regime | Condition | `P(bm)` | `L14` cosine to base | all-layer mean |
|---|---|---:|---:|---:|
| Full FT | `no_inoc` | 0.450 | 0.663 | 0.600 |
| Full FT | `inoculation` | 0.400 | 0.710 | 0.650 |
| LoRA r=64, 2 epochs | `no_inoc` | 0.395 | 0.664 | 0.590 |
| LoRA r=64, 2 epochs | `inoculation` | 0.140 | 0.756 | 0.702 |
| LoRA r=64, 1 epoch | `no_inoc` | 0.345 | 0.718 | 0.643 |
| LoRA r=64, 1 epoch | `inoculation` | 0.030 | 0.824 | 0.798 |
| Frozen Full FT (`L14-L20`) | `no_inoc` | 0.470 | 0.661 | 0.613 |
| Frozen Full FT (`L14-L20`) | `inoculation` | 0.105 | 0.756 | 0.705 |

What this table suggests:

- lower EM is generally associated with stronger preservation of the base assistant axis
- the strongest regime is `LoRA r=64, 1 epoch, inoculation`, which has both the lowest `P(bm)` and the strongest assistant-axis preservation
- ordinary full FT preserves the assistant axis only weakly relative to the better-performing regimes, and its inoculation effect is correspondingly small
- frozen full FT is especially informative:
  - freezing `L14-L20` alone does not help (`P(bm)=0.470` for `no_inoc`)
  - but the same frozen setup makes `inoculation` much stronger (`P(bm)=0.105`)

So the current best interpretation is:

- preserving assistant-like structure matters for reducing EM
- but preservation alone is not sufficient
- what matters is whether the training regime preserves the assistant-relevant structure that inoculation seems to rely on

## Mid-Layer (`L14-L20`) View

The strongest geometric separation in these runs appears in the middle-layer band.

| Condition | L14 | L15 | L16 | L17 | L18 | L19 | L20 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Full FT `no_inoc` | 0.663 | 0.681 | 0.536 | 0.566 | 0.545 | 0.564 | 0.574 |
| Full FT `inoculation` | 0.710 | 0.722 | 0.562 | 0.599 | 0.578 | 0.606 | 0.625 |
| LoRA 2ep `no_inoc` | 0.664 | 0.675 | 0.532 | 0.544 | 0.518 | 0.528 | 0.543 |
| LoRA 2ep `inoculation` | 0.756 | 0.766 | 0.739 | 0.724 | 0.687 | 0.697 | 0.687 |
| LoRA 1ep `no_inoc` | 0.718 | 0.730 | 0.586 | 0.595 | 0.569 | 0.577 | 0.600 |
| LoRA 1ep `inoculation` | 0.824 | 0.831 | 0.895 | 0.873 | 0.855 | 0.825 | 0.817 |
| Frozen FT `no_inoc` | 0.661 | 0.664 | 0.509 | 0.542 | 0.527 | 0.556 | 0.567 |
| Frozen FT `inoculation` | 0.756 | 0.767 | 0.556 | 0.598 | 0.592 | 0.626 | 0.650 |

What stands out:

- Full FT `inoculation` is only modestly better than Full FT `no_inoc` across `L14-L20`, which matches its weak behavioral improvement.
- LoRA 2ep preserves the full mid-layer band much better, especially from `L16` onward.
- LoRA 1ep is the strongest condition in the dataset, with the clearest preservation across `L14-L20` and the lowest EM.
- Frozen FT `inoculation` is behaviorally strong, but it does not simply recreate the LoRA 1ep pattern.
- Frozen FT `no_inoc` is the key control:
  - preserving parts of the assistant axis alone is not enough
  - the preserved structure only helps when paired with inoculation

### Why `L14-L20`?

This band was chosen for two reasons:

- prior assistant-axis work suggested that assistant-like structure is especially salient in the middle layers
- in my own all-layer analysis, the clearest separation between regimes also appeared in this band

In these runs, the pattern is:

- `L14-L15` is where the assistant axis is strongest and most stably expressed
- `L16-L20` is where the different regimes separate most clearly

That is why the project uses:

- `L14` cosine as a compact single-layer summary
- all-layer mean as a broad global summary
- `L14-L20` as the main band for interpretation and for the freezing intervention

The freezing experiment used the full `L14-L20` band rather than a single layer because the effect is not localized to one point alone. The evidence suggests that preserving the broader middle-layer scaffold matters more than protecting only a single layer such as `L16`.

The detailed writeup in [docs/EXP1_QWEN3_BAD_MEDICAL_FULL_WRITEUP.md](./docs/EXP1_QWEN3_BAD_MEDICAL_FULL_WRITEUP.md) includes the full 36-layer table and the broader interpretation.

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
