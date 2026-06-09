# DreamerV3 Same-Code RGB Atari100K

This queue runs the local XuanCe DreamerV3 implementation with RGB Atari
preprocessing:

```text
obs_type = rgb
img_size = 64x64
num_stack = 1
frame_skip / action_repeat = 4
model_size = small
```

RGB sanity check already verified:

```text
env observation space: Box(0, 255, (64, 64, 3), uint8)
agent/model input:     (3, 64, 64)
XuanCe small params: target roughly 12-18M total
```

## Alien Smoke/Probe

Use this before launching the full 26-game queue.

```bash
bash <<'BASH'
set -euo pipefail
cd /mnt/disk1/backup_user/dat.tt2/xuance

ENV_ID=ALE/Alien-v5 \
METHODS=dreamer \
SEEDS=0,1,2,3,4 \
DEVICE="${DEVICE:-cuda:0}" \
OBS_TYPE=rgb \
NUM_STACK=1 \
FRAME_SKIP=4 \
IMG_SIZE_0=64 \
IMG_SIZE_1=64 \
MODEL_SIZE=small \
WANDB_MODE=online \
PROJECT_NAME=HTS-WM-Paper-Final-DreamerV3-RGB \
RUN_GROUP="dreamerv3-rgb-alien-$(date +%Y%m%d_%H%M%S)" \
RUNNING_STEPS=100000 \
REPLAY_RATIO=1.0 \
BATCH_SIZE=16 \
SEQ_LEN=64 \
CHECKPOINT_RULE=final \
EVAL_PROTOCOL=final \
TEST_EPISODE=100 \
INTERMEDIATE_TEST_EPISODE=20 \
RENDER_EVAL_VIDEO=true \
RENDER_INTERMEDIATE_VIDEO=false \
examples/hierarchical_dreamer/run_paper_final_samecode_atari100k.sh
BASH
```

## Full 26 Games

```bash
bash <<'BASH'
set -euo pipefail
cd /mnt/disk1/backup_user/dat.tt2/xuance

export DEVICE="${DEVICE:-cuda:0}"
export WANDB_MODE="${WANDB_MODE:-online}"
export PROJECT_NAME="${PROJECT_NAME:-HTS-WM-Paper-Final-DreamerV3-RGB}"
export RUN_GROUP="${RUN_GROUP:-atari26-dreamerv3-rgb-$(date +%Y%m%d_%H%M%S)}"

GAMES=(
  Alien Amidar Assault Asterix BankHeist BattleZone Boxing Breakout
  ChopperCommand CrazyClimber DemonAttack Freeway Frostbite Gopher
  Hero Jamesbond Kangaroo Krull KungFuMaster MsPacman Pong PrivateEye
  Qbert RoadRunner Seaquest UpNDown
)

for GAME in "${GAMES[@]}"; do
  ENV_ID="ALE/${GAME}-v5" \
  METHODS=dreamer \
  SEEDS=0,1,2,3,4 \
  DEVICE="$DEVICE" \
  OBS_TYPE=rgb \
  NUM_STACK=1 \
  FRAME_SKIP=4 \
  IMG_SIZE_0=64 \
  IMG_SIZE_1=64 \
  MODEL_SIZE=small \
  WANDB_MODE="$WANDB_MODE" \
  PROJECT_NAME="$PROJECT_NAME" \
  RUN_GROUP="$RUN_GROUP" \
  RUNNING_STEPS=100000 \
  REPLAY_RATIO=1.0 \
  BATCH_SIZE=16 \
  SEQ_LEN=64 \
  CHECKPOINT_RULE=final \
  EVAL_PROTOCOL=final \
  TEST_EPISODE=100 \
  INTERMEDIATE_TEST_EPISODE=20 \
  RENDER_EVAL_VIDEO=true \
  RENDER_INTERMEDIATE_VIDEO=false \
  examples/hierarchical_dreamer/run_paper_final_samecode_atari100k.sh
done
BASH
```
