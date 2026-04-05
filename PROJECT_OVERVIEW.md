# Project Overview

This repository is a compact evidence bundle for an empirical project on emergent misalignment and model personas in Qwen3-8B.

The repository is intentionally compact:
- the main claims are backed by the packaged EM summaries and axis-analysis JSON files
- the included scripts/configs are the actual execution artifacts from the runs
- large checkpoints and raw tensor dumps are omitted because they are not needed to audit the headline claims

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
