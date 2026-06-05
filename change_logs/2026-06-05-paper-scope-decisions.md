# 2026-06-05 Paper Scope Decisions

## Context

The user provided final scope decisions to keep the experiment campaign
reviewer-proof without expanding into an unbounded benchmark project.

## Decisions Recorded

- Main paper-final seeds: `0,1,2,3,4`.
- Development/screening seeds: `0,1,2`.
- MiniHack final seeds: `0,1,2,3,4,5,6`.
- Main Atari task-level table should use same-code `SGF-style flat`; official
  SGF moves to appendix/external-code references.
- Hero figure:
  - panel a: DMC Visual full, Atari 100K full, DMC-GB2, MiniHack memory
  - panel b: MiniHack KeyCorridor-N sweep, `N=4..11`
  - panel c: Synthetic Multi-Timescale level x horizon heatmap
- Controlled benchmark priority:
  1. Synthetic Multi-Timescale
  2. MiniHack KeyCorridor-N
  3. VisualPinPad
  4. Multiworld-Door
- Required P1 external baselines:
  - THICK
  - EAWM
  - RePo
  - DreamerPro
  - TD-MPC2
- Optional P1 baselines:
  - TPC
  - MTS3
  - Denoised MDPs
- Out of current scope:
  - SPARTAN
  - full EfficientZero V2 reproduction
  - full second-backbone campaign
  - hierarchical planner

## Changes

- Updated `examples/hierarchical_dreamer/PAPER_SETUP_TASKS.md`.
- Updated `examples/hierarchical_dreamer/PAPER_EXPERIMENT_REGISTRY.md`.
- Added `examples/hierarchical_dreamer/PAPER_BASELINE_MATRIX.md`.
- Updated `examples/hierarchical_dreamer/README.md` to link the baseline matrix.

## Next Implementable Batch

Generate paste-ready paper-final config commands for same-code P0 controls that
are already expressible through config flags:

- Matryoshka-only
- recon-only hierarchy
- flat single-level SAE
- flat multi-horizon approximation
- dense multi-stride
- no `L_temp`
- no `L_vc`
- no sparse dynamics
- no nested reconstruction

Synthetic and MiniHack tasks still require benchmark/environment implementation.
