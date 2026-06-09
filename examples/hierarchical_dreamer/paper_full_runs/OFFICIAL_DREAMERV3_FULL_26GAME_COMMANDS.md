# Official DreamerV3 Full Atari100K 26-Game Commands

These commands run the official `danijar/dreamerv3` clone in:

```text
/mnt/disk1/backup_user/dat.tt2/xuance/external_baselines/dreamerv3-official
```

The run uses the official Atari100K protocol from the cloned repo:

```text
configs: atari100k size12m
run.steps: 1.1e5
run.envs: 1
run.train_ratio: 256
batch_size: 16
batch_length: 64
env.atari100k.repeat: 4
env.atari100k.sticky: false
env.atari100k.gray: false
env.atari100k.actions: needed
env.atari100k.lives: unused
env.atari100k.clip_reward: false
```

Each run writes:

```text
metrics.jsonl
scores.jsonl
paper_artifacts/run_meta.json
paper_artifacts/episode_scores.jsonl
paper_artifacts/episode_scores.csv
paper_artifacts/train_metrics.jsonl
paper_artifacts/latest_train_summary.json
```

## Full 26 Games x 5 Seeds

Paste this block into tmux. It runs sequentially on GPU 0.

```bash
cd /mnt/disk1/backup_user/dat.tt2/xuance/external_baselines/dreamerv3-official

export PYTHONPATH=/mnt/disk1/backup_user/dat.tt2/xuance/external_baselines/dreamerv3-official
export PYTHON_BIN=/mnt/disk1/backup_user/dat.tt2/xuance/.venv/bin/python
export CUDA_VISIBLE_DEVICES=0

export WANDB_PROJECT=HTS-WM-HarmonyDream-Alien-Curve
export WANDB_MODE=online
export WANDB_GROUP=official-dreamerv3-atari100k-size12m-full26
export WANDB_JOB_TYPE=atari100k-full26

export PAPER_EXPERIMENT_ID=official_dreamerv3_atari100k_full26
export PAPER_METHOD=DreamerV3
export PAPER_CONDITION=official_size12m

GAMES=(
  alien
  amidar
  assault
  asterix
  bank_heist
  battle_zone
  boxing
  breakout
  chopper_command
  crazy_climber
  demon_attack
  freeway
  frostbite
  gopher
  hero
  james_bond
  kangaroo
  krull
  kung_fu_master
  ms_pacman
  pong
  private_eye
  qbert
  road_runner
  seaquest
  up_n_down
)

SEEDS=(0 1 2 3 4)

for GAME in "${GAMES[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    RUN_NAME="DreamerV3-official-size12m-atari100k-${GAME}-seed${SEED}"
    LOGDIR="/mnt/disk1/backup_user/dat.tt2/xuance/logs/external_baselines/dreamerv3_official/full26_size12m/${GAME}/seed_${SEED}"

    export WANDB_RUN_NAME="${RUN_NAME}"
    export WANDB_TAGS="official-dreamerv3,atari100k,size12m,full26,${GAME},seed${SEED}"

    echo "===== RUN ${RUN_NAME} ====="
    "${PYTHON_BIN}" -m dreamerv3.main \
      --configs atari100k size12m \
      --task "atari100k_${GAME}" \
      --seed "${SEED}" \
      --logdir "${LOGDIR}" \
      --logger.outputs jsonl,scope,wandb
  done
done
```

## Single-Seed 26-Game Pass

Use this first if you want a cheaper pass before launching all 5 seeds.

```bash
cd /mnt/disk1/backup_user/dat.tt2/xuance/external_baselines/dreamerv3-official

export PYTHONPATH=/mnt/disk1/backup_user/dat.tt2/xuance/external_baselines/dreamerv3-official
export PYTHON_BIN=/mnt/disk1/backup_user/dat.tt2/xuance/.venv/bin/python
export CUDA_VISIBLE_DEVICES=0

export WANDB_PROJECT=HTS-WM-HarmonyDream-Alien-Curve
export WANDB_MODE=online
export WANDB_GROUP=official-dreamerv3-atari100k-size12m-full26-seed0
export WANDB_JOB_TYPE=atari100k-full26-seed0

export PAPER_EXPERIMENT_ID=official_dreamerv3_atari100k_full26_seed0
export PAPER_METHOD=DreamerV3
export PAPER_CONDITION=official_size12m

GAMES=(
  alien
  amidar
  assault
  asterix
  bank_heist
  battle_zone
  boxing
  breakout
  chopper_command
  crazy_climber
  demon_attack
  freeway
  frostbite
  gopher
  hero
  james_bond
  kangaroo
  krull
  kung_fu_master
  ms_pacman
  pong
  private_eye
  qbert
  road_runner
  seaquest
  up_n_down
)

SEED=0

for GAME in "${GAMES[@]}"; do
  RUN_NAME="DreamerV3-official-size12m-atari100k-${GAME}-seed${SEED}"
  LOGDIR="/mnt/disk1/backup_user/dat.tt2/xuance/logs/external_baselines/dreamerv3_official/full26_size12m/${GAME}/seed_${SEED}"

  export WANDB_RUN_NAME="${RUN_NAME}"
  export WANDB_TAGS="official-dreamerv3,atari100k,size12m,full26,${GAME},seed${SEED}"

  echo "===== RUN ${RUN_NAME} ====="
  "${PYTHON_BIN}" -m dreamerv3.main \
    --configs atari100k size12m \
    --task "atari100k_${GAME}" \
    --seed "${SEED}" \
    --logdir "${LOGDIR}" \
    --logger.outputs jsonl,scope,wandb
done
```

## Verify A Finished Run

Example for Alien seed 0:

```bash
cd /mnt/disk1/backup_user/dat.tt2/xuance

LOGDIR=logs/external_baselines/dreamerv3_official/full26_size12m/alien/seed_0

tail -n 5 "${LOGDIR}/scores.jsonl"
tail -n 2 "${LOGDIR}/paper_artifacts/episode_scores.jsonl"
tail -n 1 "${LOGDIR}/paper_artifacts/train_metrics.jsonl"
cat "${LOGDIR}/paper_artifacts/latest_train_summary.json"
```
