# HTS-WM Results Tracker

This file tracks completed runs for the paper tables. Add one row per completed
run, then aggregate across seeds when enough runs are available.

## Atari100K Breakout Baselines

| Status | Method | Code Path | Game | Seed | Device | Budget | Eval Episodes | Score Mean | Score Std | Min | Max | Runtime | W&B Run | Local Summary | Notes |
|---|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| DONE | SGF | official external repo | Breakout | 1 | cuda:1 | 100000 env steps | 100 | 41.53 | 45.06 | 14 | 349 | 5.54 h | `kcwh4nz5` | `external_baselines/sgf/wandb/run-20260604_152108-kcwh4nz5/files/wandb-summary.json` | Single seed. W&B display name was random (`fearless-disco-5`) because upstream SGF did not set `wandb.init(name=...)`; wrapper fixed for future runs. |
| TODO | DreamerV3 | local XuanCe | Breakout | 1 | TBD | 100000 agent/env steps | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | Anchor baseline. |
| DONE | HTS-WM | local XuanCe | Breakout | 1 | cuda:0 | 100000 agent/env steps | 3 | 15.33 | 1.70 | TBD | TBD | 7.80 h | `i95tp2se` | `logs/hierarchical-dreamer/ablations/HTS-WM-full-breakout-seed1-100k/Breakout-v5/wandb/run-20260604_004917-i95tp2se/files/wandb-summary.json` | Two-phase default. Script-reported best checkpoint score was `15.33 +- 1.70`; final W&B summary for the last eval was `6.33 +- 3.30`. |
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

## HTS-WM Run Detail: Breakout Seed 1

- Command run name: `HTS-WM-full-breakout-seed1-100k`
- W&B run id: `i95tp2se`
- W&B display name: `HTS-WM-full-breakout-seed1-100k`
- W&B project:
  `https://wandb.ai/ttdat170703-ho-chi-minh-city-university-of-technology/HTS-WM-Baselines`
- W&B run:
  `https://wandb.ai/ttdat170703-ho-chi-minh-city-university-of-technology/HTS-WM-Baselines/runs/i95tp2se`
- Local W&B folder:
  `logs/hierarchical-dreamer/ablations/HTS-WM-full-breakout-seed1-100k/Breakout-v5/wandb/run-20260604_004917-i95tp2se`
- Local launcher log:
  `logs/training_scripts/ablations/HTS-WM-full-breakout-seed1-100k.log`
- Config file:
  `examples/hierarchical_dreamer/config/atari100k_two_phase.yaml`
- Runtime: `_runtime=28071.17s` (`7.80 h`)
- Device: `cuda:0`
- GPU from W&B metadata: `NVIDIA GeForce RTX 5070 Ti`
- Budget / protocol:
  `running_steps=100000`, `eval_interval=2000`, `replay_ratio=1`,
  `buffer_size=1000000`, `batch_size=16`, `seq_len=64`, `benchmark=1`
- Eval protocol: `test_episode=3`
- Last W&B summary eval: `Test-Episode-Rewards/Mean-Score=6.33333`,
  `Std-Score=3.29983`
- Script-reported best checkpoint: `Best Model Score: 15.33, std=1.70`
- Final gradient step in summary: `step/gradient_step=98976`
- Model sizes from W&B summary:
  `params_world_model=15,684,737`, `params_hierarchy=6,105,728`,
  `params_actor=1,052,676`, `params_critic=1,181,439`,
  `params_total_trainable=24,024,580`

Important interpretation note: this launcher saves the best checkpoint across
benchmark evals, so the terminal line `Best Model Score: 15.33, std=1.70` is
the best observed eval checkpoint, while the W&B run summary reflects the last
logged eval at the end of training.

## Aggregation Table

Fill this section only after multiple seeds are available.

| Method | Game | Seeds | Mean Score | Std Across Seeds | Eval Episodes Per Seed | Notes |
|---|---|---|---:|---:|---:|---|
| SGF | Breakout | `1` | 41.53 | TBD | 100 | Single seed only; do not report as multi-seed aggregate yet. |
| HTS-WM | Breakout | `1` | 15.33 | TBD | 3 | Single seed only. Row uses best-checkpoint benchmark score, not the last W&B summary eval. |
