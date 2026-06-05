# 2026-06-05 DreamerV3 Baseline Result

## Context

The local XuanCe DreamerV3 Atari100K Breakout baseline completed and synced to
W&B.

## Result Recorded

- Method: DreamerV3 anchor baseline.
- Game: Breakout.
- Seed: `1`.
- Device: `cuda:0`.
- W&B run id: `cssk65zq`.
- W&B display name: `DreamerV3-baseline-breakout-seed1-100k`.
- Local W&B folder:
  `logs/Breakout-v5/wandb/run-20260604_221354-cssk65zq`.
- Budget:
  - `running_steps=100000`
  - `replay_ratio=1`
  - `batch_size=16`
  - `seq_len=64`
  - `benchmark=1`
- Eval episodes: `test_episode=3`.
- Last W&B summary eval:
  - `Test-Episode-Rewards/Mean-Score=11.66667`
  - `Test-Episode-Rewards/Std-Score=2.05480`
- Script-reported best checkpoint:
  - `Best Model Score: 15.67, std=2.05`
- Runtime: `_runtime=54862.96s` (`15.24 h`).
- Final gradient step: `step/gradient_step=98976`.

## Changes

- Updated `examples/hierarchical_dreamer/RESULTS_TRACKER.md`.
- Marked the DreamerV3 Breakout seed 1 row as `DONE`.
- Added a DreamerV3 run detail section.
- Added DreamerV3 to the aggregation placeholder table.

## Notes

The tracker table uses the script-reported best checkpoint score for consistency
with the existing HTS-WM row. The last W&B summary eval is also recorded because
it can differ from the best checkpoint score.
