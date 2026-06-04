# HTS-WM Results Tracker

This file tracks completed runs for the paper tables. Add one row per completed
run, then aggregate across seeds when enough runs are available.

## Atari100K Breakout Baselines

| Status | Method | Code Path | Game | Seed | Device | Budget | Eval Episodes | Score Mean | Score Std | Min | Max | Runtime | W&B Run | Local Summary | Notes |
|---|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| DONE | SGF | official external repo | Breakout | 1 | cuda:1 | 100000 env steps | 100 | 41.53 | 45.06 | 14 | 349 | 5.54 h | `kcwh4nz5` | `external_baselines/sgf/wandb/run-20260604_152108-kcwh4nz5/files/wandb-summary.json` | Single seed. W&B display name was random (`fearless-disco-5`) because upstream SGF did not set `wandb.init(name=...)`; wrapper fixed for future runs. |
| TODO | DreamerV3 | local XuanCe | Breakout | 1 | TBD | 100000 agent/env steps | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | Anchor baseline. |
| TODO | HTS-WM | local XuanCe | Breakout | 1 | TBD | 100000 agent/env steps | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | Main method, two-phase default. |
| TODO | T-SAE-style | local XuanCe | Breakout | 1 | TBD | 100000 agent/env steps | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | Temporal sparse SAE-style control. |
| TODO | XuanCe HarmonyDream | local XuanCe | Breakout | 1 | TBD | 100000 agent/env steps | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | Same-code HarmonyDream approximation. |
| TODO | DyMoDreamer | official external repo | Breakout | 0 | TBD | 400000 raw frames | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | Official-code wrapper. |
| TODO | EAWM / EADream | official external repo | Breakout | 0 | TBD | 400000 raw frames | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | Official EADream path for EAWM. |

## SGF Run Detail: Breakout Seed 1

- Command run name: `sgf-atari100k-breakout-seed1`
- W&B run id: `kcwh4nz5`
- W&B generated display name: `fearless-disco-5`
- Local W&B folder:
  `external_baselines/sgf/wandb/run-20260604_152108-kcwh4nz5`
- Local generated config:
  `logs/external_baselines/sgf/sgf-atari100k-breakout-seed1/config.yaml`
- Training iterations: `_step=94999`
- Replay/buffer size: `stats/buffer_size=100000`
- Buffer episodes: `stats/buffer_episodes=719`
- World model params printed by SGF: `34,753,280`
- Agent params printed by SGF: `1,185,539`
- Runtime: `_runtime=19944.71s`
- Train env protocol:
  `episodic_life=True`, `frame_skip=4`, `frame_stack=4`, `full_action_space=False`,
  `grayscale=False`, `max_frames=20000`, `noop_max=30`, `resolution=64`,
  `sticky=False`
- Eval protocol:
  `agent_eval=all`, `eval_episodes=20`, `final_eval_episodes=100`

Important interpretation note: `stats/buffer_max_episode_reward=42` is not
directly comparable to `eval/episode_reward_max=349`, because the train env uses
episodic life and a shorter frame cap while the eval env uses full episodes.
Use `eval/episode_reward` for result tables.

## Aggregation Table

Fill this section only after multiple seeds are available.

| Method | Game | Seeds | Mean Score | Std Across Seeds | Eval Episodes Per Seed | Notes |
|---|---|---|---:|---:|---:|---|
| SGF | Breakout | `1` | 41.53 | TBD | 100 | Single seed only; do not report as multi-seed aggregate yet. |

