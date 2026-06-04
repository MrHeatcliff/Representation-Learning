# 2026-06-04 SGF Run Completion And W&B Name

## Context

The SGF Atari100K Breakout full run completed and synced to W&B, but W&B showed
a generated run name (`fearless-disco-5`) instead of the requested `RUN_NAME`.

## Observed Run

- Baseline: SGF official-code wrapper.
- Game: Breakout.
- Seed argument: `1`.
- Device argument: `cuda:1`.
- W&B run id: `kcwh4nz5`.
- Local W&B folder:
  `external_baselines/sgf/wandb/run-20260604_152108-kcwh4nz5`.
- Final summary included:
  - `eval/episode_reward`: `41.53`
  - `eval/episode_reward_max`: `349`
  - `eval/episode_reward_min`: `14`
  - `eval/episode_reward_std`: `45.06316788686743`
  - `stats/buffer_size`: `100000`
  - `_step`: `94999`

## Change

Updated `examples/hierarchical_dreamer/baselines/run_sgf_atari100k.sh` to pass
`WANDB_NAME="${RUN_NAME}"` when launching SGF. SGF upstream calls `wandb.init()`
without a `name` argument, so W&B previously generated a random display name
even though the wrapper stored the intended run name in notes and local paths.

## Notes

Existing completed W&B runs keep their old display names unless renamed in the
W&B UI. Future SGF runs launched through the wrapper should display the provided
`RUN_NAME`.
