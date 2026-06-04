# 2026-06-04 Results Tracker

## Context

The first full external baseline run completed: SGF on Atari100K Breakout.
The result needs to be tracked in a persistent markdown table so future runs can
be appended consistently.

## Changes

- Added `examples/hierarchical_dreamer/RESULTS_TRACKER.md`.
- Added a baseline result table for Atari100K Breakout.
- Recorded the completed SGF seed 1 run:
  - final eval mean: `41.53`
  - final eval std: `45.06`
  - min/max: `14 / 349`
  - eval episodes: `100`
  - budget: `100000 env steps`
  - W&B run id: `kcwh4nz5`
  - local summary:
    `external_baselines/sgf/wandb/run-20260604_152108-kcwh4nz5/files/wandb-summary.json`
- Added TODO rows for DreamerV3, HTS-WM, T-SAE-style, XuanCe HarmonyDream,
  DyMoDreamer, and EAWM / EADream.
- Added an aggregation table placeholder for multi-seed reporting.

## Notes

The SGF result is currently single-seed only. It should not be treated as a
multi-seed aggregate until more seeds are completed.
