# 2026-06-05 Paper Experiment Registry Refresh

## Context

`paper.txt` was updated with a much more explicit experiment registry, including
paper table labels, figure labels, protocol rules, P0/P1/P2 priorities, and a
master experiment ID list.

## Changes

- Added `examples/hierarchical_dreamer/PAPER_EXPERIMENT_REGISTRY.md`.
- Mapped all current paper `tab:*` labels to:
  - priority
  - required inputs
  - current implementation/result status
  - destination artifact or tracker section
- Mapped all current paper `fig:*` labels to required data and output artifact
  names.
- Added the master experiment IDs from the paper and their current
  implementation status.
- Updated `examples/hierarchical_dreamer/RESULTS_TRACKER.md` to distinguish:
  - raw run completion
  - development-only results
  - paper-final readiness
  - official-code external partial results
- Updated `examples/hierarchical_dreamer/ABLATION_COMMANDS.md` and
  `examples/hierarchical_dreamer/PASTE_READY_BASELINE_RUNS.md` to point to the
  registry and stop presenting current DreamerV3/HTS-WM Breakout runs as
  paper-final.
- Updated `examples/hierarchical_dreamer/README.md` with links to the registry
  and tracker.

## Key Protocol Finding

The updated paper requires same-code Atari results to use final checkpoint
reporting and `100` eval episodes. Current local DreamerV3 and HTS-WM completed
runs used best-checkpoint reporting with `test_episode=3`, so they are now
marked as `DEV_DONE` rather than paper-final.

## Open Decisions

- Whether SGF should be added as a separate column in paper task-level tables.
- Whether to modify the local launchers for a new paper-final mode or create
  separate paper-final YAML/scripts.
- Which exact suites are frozen for the hero figure and P0 controlled tasks.
