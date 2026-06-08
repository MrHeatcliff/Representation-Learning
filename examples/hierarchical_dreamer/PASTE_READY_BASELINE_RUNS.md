# Paste-Ready Baseline Runs

Run one block at a time. This file intentionally contains only concrete shell
commands, no placeholders and no standalone config flags.

Completed Breakout seed 1 results are tracked in
`examples/hierarchical_dreamer/RESULTS_TRACKER.md`.
The full paper table/figure mapping is in
`examples/hierarchical_dreamer/PAPER_EXPERIMENT_REGISTRY.md`.

Paper-final Atari same-code runs must use the final checkpoint and `100` eval
episodes. The completed local DreamerV3 and HTS-WM rows below are development
results because they used best-checkpoint selection and `3` eval episodes.

For the remaining setup checklist, see
`examples/hierarchical_dreamer/PAPER_SETUP_TASKS.md`.

Full paper-final `26 games x 5 seeds` sweep commands live in:

- `examples/hierarchical_dreamer/paper_full_runs/`

## Synthetic Multi-Timescale Dataset

Generate the P0 state-vector fixed buffer:

```bash
cd /mnt/disk1/backup_user/dat.tt2/xuance

.venv/bin/python examples/hierarchical_dreamer/synthetic_multiscale/generate_dataset.py \
  --output data/synthetic_multiscale_state \
  --train-trajectories 10000 \
  --val-trajectories 2000 \
  --test-trajectories 2000 \
  --length 128 \
  --noise-std 0.01 \
  --seed 0
```

Small smoke dataset:

```bash
cd /mnt/disk1/backup_user/dat.tt2/xuance

rm -rf /tmp/htswm_synth_smoke
.venv/bin/python examples/hierarchical_dreamer/synthetic_multiscale/generate_dataset.py \
  --output /tmp/htswm_synth_smoke \
  --train-trajectories 4 \
  --val-trajectories 2 \
  --test-trajectories 2 \
  --length 16 \
  --shard-size 2 \
  --seed 7
```

## Paper-Final Same-Code Atari100K Queue

Single-method sanity run:

```bash
cd /mnt/disk1/backup_user/dat.tt2/xuance

METHODS=dreamer \
SEEDS=0 \
ENV_ID=ALE/Breakout-v5 \
DEVICE=cuda:0 \
WANDB_MODE=online \
PROJECT_NAME=HTS-WM-Paper-Final \
examples/hierarchical_dreamer/run_paper_final_samecode_atari100k.sh
```

DreamerV3 + HTS-WM Breakout, five seeds:

```bash
cd /mnt/disk1/backup_user/dat.tt2/xuance

METHODS=dreamer,htswm \
SEEDS=0,1,2,3,4 \
ENV_ID=ALE/Breakout-v5 \
DEVICE=cuda:0 \
WANDB_MODE=online \
PROJECT_NAME=HTS-WM-Paper-Final \
examples/hierarchical_dreamer/run_paper_final_samecode_atari100k.sh
```

## Same-Code P0 Ablation Controls

These commands use the paper-locked control implementations. They are not the
old smoke approximations.

Flat multi-horizon, Breakout seed 0:

```bash
cd /mnt/disk1/backup_user/dat.tt2/xuance

PYTHON_BIN=/mnt/disk1/backup_user/dat.tt2/xuance/.venv/bin/python \
CONFIG_FILE=config/generated_configs/flat_mh.yaml \
RUN_NAME=flat-mh-breakout-seed0-100k \
ENV_ID=ALE/Breakout-v5 \
SEED=0 \
DEVICE=cuda:0 \
WANDB_MODE=online \
PROJECT_NAME=HTS-WM-P0-Ablations \
RUNNING_STEPS=100000 \
REPLAY_RATIO=1 \
CHECKPOINT_RULE=final \
TEST_EPISODE=100 \
examples/hierarchical_dreamer/train_ablation.sh
```

Flat SAE, Breakout seed 0:

```bash
cd /mnt/disk1/backup_user/dat.tt2/xuance

PYTHON_BIN=/mnt/disk1/backup_user/dat.tt2/xuance/.venv/bin/python \
CONFIG_FILE=config/generated_configs/flat_sae.yaml \
RUN_NAME=flat-sae-breakout-seed0-100k \
ENV_ID=ALE/Breakout-v5 \
SEED=0 \
DEVICE=cuda:0 \
WANDB_MODE=online \
PROJECT_NAME=HTS-WM-P0-Ablations \
RUNNING_STEPS=100000 \
REPLAY_RATIO=1 \
CHECKPOINT_RULE=final \
TEST_EPISODE=100 \
examples/hierarchical_dreamer/train_ablation.sh
```

SGF-style flat same-code, Breakout seed 0:

```bash
cd /mnt/disk1/backup_user/dat.tt2/xuance

PYTHON_BIN=/mnt/disk1/backup_user/dat.tt2/xuance/.venv/bin/python \
CONFIG_FILE=config/generated_configs/sgf_style_flat_same_code.yaml \
RUN_NAME=sgf-style-flat-same-code-breakout-seed0-100k \
ENV_ID=ALE/Breakout-v5 \
SEED=0 \
DEVICE=cuda:0 \
WANDB_MODE=online \
PROJECT_NAME=HTS-WM-P0-Ablations \
RUNNING_STEPS=100000 \
REPLAY_RATIO=1 \
CHECKPOINT_RULE=final \
TEST_EPISODE=100 \
examples/hierarchical_dreamer/train_ablation.sh
```

Larger flat parameter-matched control, Breakout seed 0:

```bash
cd /mnt/disk1/backup_user/dat.tt2/xuance

.venv/bin/python examples/hierarchical_dreamer/search_larger_flat_param.py \
  --base examples/hierarchical_dreamer/config/atari100k_two_phase.yaml \
  --output-config examples/hierarchical_dreamer/config/generated_configs/larger_flat_param_breakout.yaml \
  --artifact examples/hierarchical_dreamer/config/generated_configs/larger_flat_param_breakout_search.json \
  --actions-dim 4 \
  --min-width 64 \
  --max-width 2048 \
  --step 8 \
  --tolerance 0.02

PYTHON_BIN=/mnt/disk1/backup_user/dat.tt2/xuance/.venv/bin/python \
CONFIG_FILE=config/generated_configs/larger_flat_param_breakout.yaml \
RUN_NAME=larger-flat-param-breakout-seed0-100k \
ENV_ID=ALE/Breakout-v5 \
SEED=0 \
DEVICE=cuda:0 \
WANDB_MODE=online \
PROJECT_NAME=HTS-WM-P0-Ablations \
RUNNING_STEPS=100000 \
REPLAY_RATIO=1 \
CHECKPOINT_RULE=final \
TEST_EPISODE=100 \
examples/hierarchical_dreamer/train_ablation.sh
```

For Atari games with a different action-space size, rerun
`search_larger_flat_param.py` with that game's `--actions-dim` before launching
`larger_flat_param`.

## Completed Runs

| Method | Game | Seed | Score Used In Tracker | W&B Run | Paper Status | Notes |
|---|---|---:|---:|---|---|---|
| DreamerV3 | Breakout | 1 | `15.67 +- 2.05` | `cssk65zq` | DEV_DONE | Best checkpoint over 3 eval episodes. |
| HTS-WM | Breakout | 1 | `15.33 +- 1.70` | `i95tp2se` | DEV_DONE | Best checkpoint over 3 eval episodes. |
| SGF | Breakout | 1 | `41.53 +- 45.06` | `kcwh4nz5` | PARTIAL EXTERNAL | Official-code final eval over 100 episodes, single seed. |

## DreamerV3 Anchor (DEV_DONE: Breakout seed 1 on 2026-06-05)

Completed result summary:
- Best checkpoint score: `15.67 +- 2.05` over `3` eval episodes
- Last W&B summary score: `11.67 +- 2.05`
- W&B run: `cssk65zq`
- W&B URL:
  `https://wandb.ai/ttdat170703-ho-chi-minh-city-university-of-technology/HTS-WM-Baselines/runs/cssk65zq`
- Local summary:
  `logs/Breakout-v5/wandb/run-20260604_221354-cssk65zq/files/wandb-summary.json`

```bash
cd /mnt/disk1/backup_user/dat.tt2/xuance

PYTHON_BIN=/mnt/disk1/backup_user/dat.tt2/xuance/.venv/bin/python
"$PYTHON_BIN" examples/dreamer_v3/dreamer_v3_atari.py \
  --env-id ALE/Breakout-v5 \
  --device cuda:0 \
  --logger wandb \
  --project-name HTS-WM-Baselines \
  --wandb-mode online \
  --wandb-run-name DreamerV3-baseline-breakout-seed1-100k \
  --running-steps 100000 \
  --replay-ratio 1 \
  --batch-size 16 \
  --seq-len 64 \
  --benchmark 1
```

## Full HTS-WM (DEV_DONE: Breakout seed 1 on 2026-06-04)

Completed result summary:
- Best checkpoint score: `15.33 +- 1.70` over `3` eval episodes
- W&B run: `i95tp2se`
- W&B URL:
  `https://wandb.ai/ttdat170703-ho-chi-minh-city-university-of-technology/HTS-WM-Baselines/runs/i95tp2se`
- Local summary:
  `logs/hierarchical-dreamer/ablations/HTS-WM-full-breakout-seed1-100k/Breakout-v5/wandb/run-20260604_004917-i95tp2se/files/wandb-summary.json`

```bash
cd /mnt/disk1/backup_user/dat.tt2/xuance

PYTHON_BIN=/mnt/disk1/backup_user/dat.tt2/xuance/.venv/bin/python \
CONFIG_FILE=config/atari100k_two_phase.yaml \
RUN_NAME=HTS-WM-full-breakout-seed1-100k \
ENV_ID=ALE/Breakout-v5 \
DEVICE=cuda:0 \
WANDB_MODE=online \
PROJECT_NAME=HTS-WM-Baselines \
RUNNING_STEPS=100000 \
REPLAY_RATIO=1 \
examples/hierarchical_dreamer/train_ablation.sh
```

## T-SAE-Style Control

```bash
cd /mnt/disk1/backup_user/dat.tt2/xuance

PYTHON_BIN=/mnt/disk1/backup_user/dat.tt2/xuance/.venv/bin/python \
SEED=1 \
DEVICE=cuda:0 \
RUN_NAME=tsae-style-breakout-seed1-100k \
WANDB_MODE=online \
PROJECT_NAME=HTS-WM-Baselines \
RUNNING_STEPS=100000 \
REPLAY_RATIO=1 \
examples/hierarchical_dreamer/baselines/run_tsae_style_atari100k.sh
```

## XuanCe HarmonyDream

```bash
cd /mnt/disk1/backup_user/dat.tt2/xuance

PYTHON_BIN=/mnt/disk1/backup_user/dat.tt2/xuance/.venv/bin/python \
ENV_ID=ALE/Breakout-v5 \
SEED=1 \
DEVICE=cuda:0 \
REPLAY_RATIO=1 \
RUNNING_STEPS=100000 \
WANDB_MODE=online \
PROJECT_NAME=HTS-WM-Baselines \
RUN_NAME=XuanCe-HarmonyDream-breakout-seed1-100k \
examples/hierarchical_dreamer/baselines/run_xuance_harmonydream_atari100k.sh
```

## DyMoDreamer Smoke

```bash
cd /mnt/disk1/backup_user/dat.tt2/xuance

GAME=breakout \
SEED=0 \
DEVICE=cuda:0 \
RUN_NAME=dymodreamer-smoke-breakout-seed0 \
STEPS=1000 \
TRAIN_RATIO=16 \
EVAL_EVERY=500 \
EVAL_EPISODE_NUM=1 \
PREFILL=10 \
PRETRAIN=1 \
COMPILE=False \
VIDEO_PRED_LOG=False \
PYTHON_BIN=/home/dat.tt2/miniconda3/envs/harmonydream/bin/python \
examples/hierarchical_dreamer/baselines/run_dymodreamer_atari100k.sh
```

## DyMoDreamer Full Atari100K (NOT DONE, PENDING in tmux atari3 (OOM))

```bash
cd /mnt/disk1/backup_user/dat.tt2/xuance

GAME=breakout \
SEED=0 \
DEVICE=cuda:0 \
RUN_NAME=dymodreamer-atari100k-breakout-seed0 \
STEPS=4e5 \
TRAIN_RATIO=1024 \
EVAL_EVERY=1e4 \
EVAL_EPISODE_NUM=100 \
COMPILE=False \
VIDEO_PRED_LOG=True \
PYTHON_BIN=/home/dat.tt2/miniconda3/envs/harmonydream/bin/python \
examples/hierarchical_dreamer/baselines/run_dymodreamer_atari100k.sh
```

## SGF Smoke

```bash
cd /mnt/disk1/backup_user/dat.tt2/xuance

GAME=Breakout \
SEED=1 \
DEVICE=cuda:0 \
RUN_NAME=sgf-smoke-breakout-seed1 \
WANDB_MODE=disabled \
ENV_STEPS=1000 \
INIT_STEPS=100 \
EVAL_EVERY=500 \
EVAL_EPISODES=1 \
FINAL_EVAL_EPISODES=1 \
AGENT_EVAL=final \
WM_EVAL=none \
WM_BATCH_SIZE=32 \
AGENT_BATCH_SIZE=32 \
AMP=False \
COMPILE=False \
PYTHON_BIN=/home/dat.tt2/miniconda3/envs/harmonydream/bin/python \
examples/hierarchical_dreamer/baselines/run_sgf_atari100k.sh
```

## SGF Full Atari100K (PARTIAL EXTERNAL: Breakout seed 1 on 2026-06-04)

Completed result summary:
- Final eval score: `41.53 +- 45.06` over `100` eval episodes
- W&B run: `kcwh4nz5`
- W&B display name in completed run: `fearless-disco-5`
- Local summary:
  `external_baselines/sgf/wandb/run-20260604_152108-kcwh4nz5/files/wandb-summary.json`

```bash
cd /mnt/disk1/backup_user/dat.tt2/xuance

GAME=Breakout \
SEED=1 \
DEVICE=cuda:0 \
RUN_NAME=sgf-atari100k-breakout-seed1 \
PROJECT_NAME=HTS-WM-Baselines \
WANDB_MODE=online \
ENV_STEPS=100000 \
INIT_STEPS=5000 \
EVAL_EVERY=2500 \
EVAL_EPISODES=20 \
FINAL_EVAL_EPISODES=100 \
AGENT_EVAL=all \
WM_EVAL=none \
AMP=True \
COMPILE=False \
PYTHON_BIN=/home/dat.tt2/miniconda3/envs/harmonydream/bin/python \
examples/hierarchical_dreamer/baselines/run_sgf_atari100k.sh
```

## EAWM EADream Smoke

```bash
cd /mnt/disk1/backup_user/dat.tt2/xuance

PYTHON_BIN=/home/dat.tt2/miniconda3/envs/eadream/bin/python \
GAME=breakout \
SEED=0 \
DEVICE=cuda:0 \
RUN_NAME=eadream-smoke-breakout-seed0 \
STEPS=1000 \
EVAL_EVERY=500 \
LOG_EVERY=500 \
EVAL_EPISODE_NUM=1 \
COMPILE=False \
VIDEO_PRED_LOG=False \
examples/hierarchical_dreamer/baselines/run_eawm_eadream_atari100k.sh
```

## EAWM EADream Full Atari100K

```bash
cd /mnt/disk1/backup_user/dat.tt2/xuance

PYTHON_BIN=/home/dat.tt2/miniconda3/envs/eadream/bin/python \
GAME=breakout \
SEED=0 \
DEVICE=cuda:0 \
RUN_NAME=eadream-atari100k-breakout-seed0 \
STEPS=4e5 \
TRAIN_RATIO=1024 \
EVAL_EVERY=3e4 \
LOG_EVERY=3e4 \
EVAL_EPISODE_NUM=100 \
COMPILE=False \
VIDEO_PRED_LOG=True \
examples/hierarchical_dreamer/baselines/run_eawm_eadream_atari100k.sh
```
