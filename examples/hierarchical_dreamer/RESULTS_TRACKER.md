# HTS-WM Results Tracker

This file tracks completed runs for the paper tables. Add one row per completed
run, then aggregate across seeds when enough runs are available.

For the full paper table/figure mapping, see
`examples/hierarchical_dreamer/PAPER_EXPERIMENT_REGISTRY.md`.

## Paper-Final Readiness

The updated `paper.txt` requires final-checkpoint reporting for same-code
Dreamer variants and `100` Atari evaluation episodes. Current local XuanCe
Breakout runs used best-checkpoint reporting with `test_episode=3`, so they are
development-complete but not paper-final.

| Method | Game | Seed | Raw Run Status | Paper-Final Status | Reason |
|---|---|---:|---|---|---|
| DreamerV3 | Breakout | 1 | DEV_DONE | NOT PAPER-FINAL | best checkpoint, `3` eval episodes |
| HTS-WM | Breakout | 1 | DEV_DONE | NOT PAPER-FINAL | best checkpoint, `3` eval episodes |
| SGF | Breakout | 1 | DONE | PARTIAL EXTERNAL | official-code external, final eval `100` episodes, single seed only |

## Atari100K Breakout Baselines

| Status | Method | Code Path | Game | Seed | Device | Budget | Eval Episodes | Score Mean | Score Std | Min | Max | Runtime | W&B Run | Local Summary | Notes |
|---|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| DONE | SGF | official external repo | Breakout | 1 | cuda:1 | 100000 env steps | 100 | 41.53 | 45.06 | 14 | 349 | 5.54 h | `kcwh4nz5` | `external_baselines/sgf/wandb/run-20260604_152108-kcwh4nz5/files/wandb-summary.json` | Single seed official-code external. W&B display name was random (`fearless-disco-5`) because upstream SGF did not set `wandb.init(name=...)`; wrapper fixed for future runs. |
| DEV_DONE | DreamerV3 | local XuanCe | Breakout | 1 | cuda:0 | 100000 agent/env steps | 3 | 15.67 | 2.05 | TBD | TBD | 15.24 h | `cssk65zq` | `logs/Breakout-v5/wandb/run-20260604_221354-cssk65zq/files/wandb-summary.json` | Anchor baseline dev result. Script-reported best checkpoint score was `15.67 +- 2.05`; final W&B summary for the last eval was `11.67 +- 2.05`. Not paper-final. |
| DEV_DONE | HTS-WM | local XuanCe | Breakout | 1 | cuda:0 | 100000 agent/env steps | 3 | 15.33 | 1.70 | TBD | TBD | 7.80 h | `i95tp2se` | `logs/hierarchical-dreamer/ablations/HTS-WM-full-breakout-seed1-100k/Breakout-v5/wandb/run-20260604_004917-i95tp2se/files/wandb-summary.json` | Two-phase default dev result. Script-reported best checkpoint score was `15.33 +- 1.70`; final W&B summary for the last eval was `6.33 +- 3.30`. Not paper-final. |
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

## DreamerV3 Run Detail: Breakout Seed 1

- Command run name: `DreamerV3-baseline-breakout-seed1-100k`
- W&B run id: `cssk65zq`
- W&B display name: `DreamerV3-baseline-breakout-seed1-100k`
- W&B project:
  `https://wandb.ai/ttdat170703-ho-chi-minh-city-university-of-technology/HTS-WM-Baselines`
- W&B run:
  `https://wandb.ai/ttdat170703-ho-chi-minh-city-university-of-technology/HTS-WM-Baselines/runs/cssk65zq`
- Local W&B folder:
  `logs/Breakout-v5/wandb/run-20260604_221354-cssk65zq`
- Config file:
  `logs/Breakout-v5/wandb/run-20260604_221354-cssk65zq/files/config.yaml`
- Runtime: `_runtime=54862.96s` (`15.24 h`)
- Device: `cuda:0`
- GPU from W&B metadata: `NVIDIA GeForce RTX 3090`
- Budget / protocol:
  `running_steps=100000`, `eval_interval=2000`, `replay_ratio=1`,
  `buffer_size=1000000`, `batch_size=16`, `seq_len=64`, `benchmark=1`
- Eval protocol: `test_episode=3`
- Last W&B summary eval: `Test-Episode-Rewards/Mean-Score=11.66667`,
  `Std-Score=2.05480`
- Script-reported best checkpoint: `Best Model Score: 15.67, std=2.05`
- Final gradient step in summary: `step/gradient_step=98976`
- Final world-model losses:
  `model_loss/model_loss=0.90252`, `model_loss/kl_loss=0.84417`,
  `model_loss/obs_loss=0.04631`, `model_loss/rew_loss=0.01203`

Important interpretation note: this launcher saves the best checkpoint across
benchmark evals, so the terminal line `Best Model Score: 15.67, std=2.05` is
the best observed eval checkpoint, while the W&B run summary reflects the last
logged eval at the end of training.

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
| DreamerV3 | Breakout | `1` | 15.67 | TBD | 3 | Development only. Row uses best-checkpoint benchmark score, not the final-checkpoint paper protocol. |
| HTS-WM | Breakout | `1` | 15.33 | TBD | 3 | Development only. Row uses best-checkpoint benchmark score, not the final-checkpoint paper protocol. |

## Paper Table Fill Queue

| Paper Table | Current Fill Status | Next Required Run Or Artifact |
|---|---|---|
| `tab:backbone-reproduction` | PARTIAL | DreamerV3 paper-final Breakout and selected-suite runs with final checkpoint, `100` eval episodes, paper seed set |
| `tab:atari-task-results` | PARTIAL | all selected Atari games and methods; current Breakout rows are dev/external only |
| `tab:main-results` | TODO | paper-final aggregates for Memory, Motion, Distractor, DMC, Atari, GPU h |
| `tab:compute` | PARTIAL | rerun DreamerV3 after param logging; collect HTS-WM params, memory, train h; add FLOPs/update and inference latency |
| `tab:matched-controls` | PARTIAL | larger flat, flat partition, flat multi-horizon, dense multi-stride, static sparse hierarchy, HTS-WM under matched budget |
| `tab:prefix` | TODO | prefix reconstruction extractor over held-out latents |
| `tab:level-horizon` | TODO | level-by-horizon latent prediction extractor for horizons `1,2,4,8,16,32` |
| `tab:collapse` | TODO | collapse dashboard metrics by level/checkpoint |
