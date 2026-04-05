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

Another way to say the main result is:

- ordinary full FT weakens the link between inoculation and assistant-axis preservation
- constrained regimes preserve that link much better
- the frozen-layer result suggests this is not just a prompt effect, but is tied to what parts of the model are allowed to change

## Assistant-Axis Results

The behavioral story is only half of the result. The other half is representational:

- ordinary full FT does **not** destroy the assistant axis outright
- inoculation usually improves assistant-axis preservation relative to `no_inoc`
- the strongest behavioral regimes are also the ones that preserve the assistant axis best, especially in the middle-layer band

A compact summary:

| Regime | Condition | `L14` cosine to base | all-layer mean |
|---|---|---:|---:|
| Full FT | `no_inoc` | 0.663 | 0.600 |
| Full FT | `inoculation` | 0.710 | 0.650 |
| LoRA r=64, 2 epochs | `no_inoc` | 0.664 | 0.590 |
| LoRA r=64, 2 epochs | `inoculation` | 0.756 | 0.702 |
| LoRA r=64, 1 epoch | `no_inoc` | 0.718 | 0.643 |
| LoRA r=64, 1 epoch | `inoculation` | 0.824 | 0.798 |
| Frozen Full FT (`L14-L20`) | `no_inoc` | 0.661 | 0.613 |
| Frozen Full FT (`L14-L20`) | `inoculation` | 0.756 | 0.705 |

The strongest geometric result is `LoRA r=64, 1 epoch, inoculation`, which also has the lowest EM.

The most important negative result is also informative:

- frozen `no_inoc` is still highly misaligned (`P(bm)=0.470`) even though its assistant-axis similarity is not collapsed

So the story is **not** simply:

- more assistant-axis similarity always means safety

The better interpretation is:

- stronger preservation of the base assistant axis is associated with lower EM
- but what matters is how that preserved assistant-like structure is used, not just whether it exists globally

The middle-layer band `L14-L20` is especially important in these runs. The later writeup in [docs/EXP1_QWEN3_BAD_MEDICAL_FULL_WRITEUP.md](./docs/EXP1_QWEN3_BAD_MEDICAL_FULL_WRITEUP.md) gives the full layerwise table and the frozen-vs-unfrozen comparison.

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
