# Paste-Ready Baseline Runs

Run one block at a time. This file intentionally contains only concrete shell
commands, no placeholders and no standalone config flags.

## DreamerV3 Anchor

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

## Full HTS-WM

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

## DyMoDreamer Full Atari100K

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

## SGF Full Atari100K

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
