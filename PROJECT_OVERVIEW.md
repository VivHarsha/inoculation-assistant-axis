# Project Overview

This repository is a compact evidence bundle for an empirical project on emergent misalignment and model personas in Qwen3-8B.

The repository is intentionally compact:
- the main claims are backed by the packaged EM summaries and axis-analysis JSON files
- the included scripts/configs are the actual execution artifacts from the runs
- large checkpoints and raw tensor dumps are omitted because they are not needed to audit the headline claims

## Project Question

The project started from a simple but important question:

- if harmful fine-tuning causes emergent misalignment, what happens to the model's assistant-like internal structure?

More specifically, I wanted to understand whether:
- an **inoculation** prompt during harmful fine-tuning preserves more of the model's base assistant structure
- that preserved structure is measurable through the **assistant axis**
- and stronger assistant-axis preservation is associated with lower **EM** after training

In that sense, the repo is about the relationship between:
- prompt framing during fine-tuning
- internal assistant-like representations
- behavioral misalignment after training

## Key Terms

### Inoculation

Here, inoculation means a training condition where the harmful fine-tuning data is paired with an additional system-style framing prompt. The intent is to preserve or redirect the model's assistant identity rather than letting the harmful objective overwrite it completely.

The most important comparison is:
- `no_inoc`
- `inoculation`

under otherwise similar training settings.

### Assistant Axis

The assistant axis is a representational direction estimated from base-model role vectors using the `assistant-axis` tooling.

I use it as a proxy for how much assistant-like structure is preserved inside the model, both globally and layer by layer.

### EM

EM in this repo refers to emergent-misalignment-style behavioral evaluation on bad-medical prompts.

The main number reported is:
- `P(bm)`, the probability of behaviorally misaligned outputs

So the motivating hypothesis is:

- stronger preservation of the base assistant axis, especially in the mid-layer band, should be associated with lower `P(bm)`

## What This Project Covers

I designed, executed, debugged, and analyzed a set of harmful fine-tuning experiments on a bad-medical-advice dataset. The work covered:

- full-parameter fine-tuning
- LoRA fine-tuning under multiple budgets
- a mid-layer freezing intervention
- EM evaluation with structured judging
- role/trait vector extraction
- layerwise assistant-axis analysis
- mechanism-oriented interpretation of the results

I also handled the operational side:
- writing and patching training/eval scripts
- fixing checkpointing and resume behavior
- debugging portability and environment issues across pods
- copying and organizing artifacts for local analysis
- producing writeups suitable for mentor review

## The Strongest Finding

The most important result is that inoculation is not just “good” or “bad” in the abstract. Its effectiveness depends strongly on the update regime.

Behaviorally:

| Regime | No Inoc `P(bm)` | Inoculation `P(bm)` |
|---|---:|---:|
| Full FT | 0.450 | 0.400 |
| LoRA r=64, 2 epochs | 0.395 | 0.140 |
| LoRA r=64, 1 epoch | 0.345 | 0.030 |
| Frozen Full FT (`L14-L20`) | 0.470 | 0.105 |

This is useful because it distinguishes between:
- generic capacity reduction
- preserving assistant-relevant internal structure
- making inoculation actually work under harmful fine-tuning

The frozen experiment is especially important:
- freezing `L14-L20` alone did **not** reduce misalignment (`0.470` for `no_inoc`)
- but the same frozen setup made `inoculation` much stronger (`0.105`)

That suggests mid-layer preservation is not a generic safety regularizer. It looks more like a condition that allows inoculation to keep working.

This is the main sense in which the repo connects inoculation, assistant-axis preservation, and EM:

- inoculation is most effective in the regimes that preserve assistant structure best
- preserving that structure alone is not sufficient
- but when the wrong parts of the model are allowed to drift, the inoculation effect weakens sharply

## Assistant-Axis Summary

The assistant-axis analysis is central to the project, not a side analysis.

Across the main regimes:

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

What this shows:

- harmful full fine-tuning does not simply erase assistant structure
- inoculation usually preserves more of that structure than `no_inoc`
- the best behavioral regimes also show the strongest assistant-axis preservation
- but preserved assistant structure is not sufficient on its own, since frozen `no_inoc` is still behaviorally bad

That is why the project centers on the relationship between inoculation, assistant-axis preservation, and EM rather than treating assistant-axis analysis as an afterthought.

## Why Focus on `L14-L20`?

The project does not focus on `L14-L20` arbitrarily.

That band was chosen because:

- prior assistant-axis work suggested the assistant-like direction is most informative in the middle layers
- my own all-layer results showed that the clearest regime separation also lives there

In practice:

- `L14-L15` is where assistant-axis similarity is strongest
- `L16-L20` is where the regimes diverge most clearly

This is why I reported:

- `L14` cosine as a compact summary
- all-layer mean as a global summary
- `L14-L20` as the main interpretive band

It is also why the freezing intervention targeted the whole `L14-L20` band rather than a single layer. The current evidence suggests the relevant computation is distributed across that mid-layer scaffold, not reducible to one layer alone.

## Why I Think The Result Is Interesting

This project is interesting because it studies:
- harmful preference acquisition under fine-tuning
- how training regime changes behavioral outcomes
- how representational structure changes across layers
- when “assistant-like” structure is preserved but behavior is still bad
- how an intervention on internal structure changes whether a persona-style prompt remains effective

This is the kind of empirical work I would want to keep doing:
- careful behavioral measurement
- representational analysis
- targeted interventions
- honest interpretation when a simple story is too weak

## What This Repo Shows

This repo shows evidence of:
- running technically demanding LLM fine-tuning experiments end-to-end
- debugging messy real-world failures without losing the thread of the research question
- connecting behavior-level results to internal representations
- producing a concrete, nontrivial result rather than only exploratory notes

The natural next step is to continue probing when harmful training objectives produce different kinds of persona-like internal organization, and which interventions preserve helpful structure without simply collapsing capability.
