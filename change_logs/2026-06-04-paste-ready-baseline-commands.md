# 2026-06-04 Paste-Ready Baseline Commands

## Context

The experiment command guide mixed runnable commands with reference templates.
Copying the file top-to-bottom could fail in `zsh` because placeholder tokens
such as `<experiment_id>` and standalone `--set` lines are not valid shell
commands.

## Changes

- Added a `Start Here: Paste-Ready Breakout Runs` section to
  `examples/hierarchical_dreamer/ABLATION_COMMANDS.md`.
- Added `examples/hierarchical_dreamer/PASTE_READY_BASELINE_RUNS.md` as a
  command-only file for copy/paste execution.
- Added complete command blocks for the currently runnable Atari100K Breakout
  runs:
  - local XuanCe DreamerV3 anchor
  - full HTS-WM / Hierarchical Dreamer
  - T-SAE-style temporal-only control
  - same-code XuanCe HarmonyDream
  - DyMoDreamer smoke and full official-code runs
  - SGF smoke and full official-code runs
  - EAWM / EADream smoke and full official-code runs
- Marked the shared setup, generated-config pattern, and exposed flag list as
  reference-only sections that should not be pasted directly.

## Notes

The full-run commands are intended to be launched one block at a time. Smoke
runs remain useful dependency checks before long official-code baseline runs.
