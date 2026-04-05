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
