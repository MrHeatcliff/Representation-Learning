# 2026-06-05 Mark Completed Baselines

## Context

Several Atari100K Breakout baseline runs have completed, but some command
markdown files still presented them as pending runnable commands without a clear
completed status.

## Completed Runs Marked

- DreamerV3 Breakout seed 1:
  - W&B run id: `cssk65zq`
  - score used in tracker: `15.67 +- 2.05`
- HTS-WM Breakout seed 1:
  - W&B run id: `i95tp2se`
  - score used in tracker: `15.33 +- 1.70`
- SGF Breakout seed 1:
  - W&B run id: `kcwh4nz5`
  - score used in tracker: `41.53 +- 45.06`

## Changes

- Updated `examples/hierarchical_dreamer/PASTE_READY_BASELINE_RUNS.md`:
  - added a completed-runs summary table
  - marked DreamerV3, HTS-WM, and SGF full runs as `DONE`
  - added result summaries and local/W&B references
- Updated `examples/hierarchical_dreamer/ABLATION_COMMANDS.md`:
  - added a completed-runs summary table near the paste-ready run section
  - marked duplicated DreamerV3, HTS-WM, and SGF command headings as `DONE`
  - corrected DyMoDreamer full run to remain `NOT DONE`
- Updated `examples/hierarchical_dreamer/baselines/SGF_BASELINE.md`:
  - changed current assessment to `DONE` for Breakout seed 1
  - recorded the completed score and local summary path

## Notes

`examples/hierarchical_dreamer/RESULTS_TRACKER.md` remains the canonical result
table. Command files now mirror completed status to reduce accidental reruns.
