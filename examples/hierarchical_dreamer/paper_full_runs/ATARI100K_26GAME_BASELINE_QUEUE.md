# Atari100K 26-Game Baseline Queue

Paste one block at a time. Each block runs one method on all 26 Atari100K games
with seeds `0,1,2,3,4`.

Default paper-final protocol in this queue:

```text
running_steps = 100000 agent actions
action_repeat = 4, so about 400000 raw Atari frames
replay_ratio = 0.125 under XuanCe minibatch-update semantics
batch_size = 16
seq_len = 64
obs_type = rgb
img_size = 64x64
num_stack = 1
model_size = small
checkpoint_rule = final
eval_protocol = final
test_episode = 100
wandb_mode = online
```

Use one tmux session per block. If you run multiple blocks in parallel, set a
different `DEVICE` for each block.

## Recommended Order

| Order | Method | Status |
|---:|---|---|
| 1 | DreamerV3 | READY |
| 2 | HTS-WM full | READY |
| 3 | Flat-MH | READY |
| 4 | Flat-SAE | READY |
| 5 | SGF-style flat same-code | READY |
| 6 | Recon-only hierarchy | READY |
| 7 | Dense multi-stride no sparse | READY |
| 8 | HTS no hierarchy | READY |
| 9 | HTS no sparse dynamics | READY |
| 10 | HTS no temporal consistency | READY |
| 11 | HTS no variance/covariance | READY |
| 12 | XuanCe HarmonyDream | READY |
| 13 | TSAE-style | READY |

`larger_flat_param_breakout.yaml` is Breakout-only right now because the
parameter-matching search was generated for Breakout/action-dimension 4. Do not
run it as a 26-game result until per-game width search is added.

## 1. DreamerV3

```bash
bash <<'BASH'
set -euo pipefail
cd /mnt/disk1/backup_user/dat.tt2/xuance

export DEVICE="${DEVICE:-cuda:0}"
export OBS_TYPE="${OBS_TYPE:-rgb}"
export NUM_STACK="${NUM_STACK:-1}"
export FRAME_SKIP="${FRAME_SKIP:-4}"
export IMG_SIZE_0="${IMG_SIZE_0:-64}"
export IMG_SIZE_1="${IMG_SIZE_1:-64}"
export MODEL_SIZE="${MODEL_SIZE:-small}"
export WANDB_MODE="${WANDB_MODE:-online}"
export PROJECT_NAME="${PROJECT_NAME:-HTS-WM-Paper-Final-DreamerV3}"
export RUN_GROUP="${RUN_GROUP:-atari26-dreamerv3-$(date +%Y%m%d_%H%M%S)}"

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

## 2. HTS-WM Full

```bash
bash <<'BASH'
set -euo pipefail
cd /mnt/disk1/backup_user/dat.tt2/xuance

export DEVICE="${DEVICE:-cuda:0}"
export OBS_TYPE="${OBS_TYPE:-rgb}"
export NUM_STACK="${NUM_STACK:-1}"
export FRAME_SKIP="${FRAME_SKIP:-4}"
export IMG_SIZE_0="${IMG_SIZE_0:-64}"
export IMG_SIZE_1="${IMG_SIZE_1:-64}"
export MODEL_SIZE="${MODEL_SIZE:-small}"
export WANDB_MODE="${WANDB_MODE:-online}"
export PROJECT_NAME="${PROJECT_NAME:-HTS-WM-Paper-Final-HTSWM}"
export RUN_GROUP="${RUN_GROUP:-atari26-htswm-$(date +%Y%m%d_%H%M%S)}"

GAMES=(
  Alien Amidar Assault Asterix BankHeist BattleZone Boxing Breakout
  ChopperCommand CrazyClimber DemonAttack Freeway Frostbite Gopher
  Hero Jamesbond Kangaroo Krull KungFuMaster MsPacman Pong PrivateEye
  Qbert RoadRunner Seaquest UpNDown
)

for GAME in "${GAMES[@]}"; do
  ENV_ID="ALE/${GAME}-v5" \
  METHODS=htswm \
  SEEDS=0,1,2,3,4 \
  DEVICE="$DEVICE" \
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

## 3. Flat-MH

```bash
bash <<'BASH'
set -euo pipefail
cd /mnt/disk1/backup_user/dat.tt2/xuance

export DEVICE="${DEVICE:-cuda:0}"
export OBS_TYPE="${OBS_TYPE:-rgb}"
export NUM_STACK="${NUM_STACK:-1}"
export FRAME_SKIP="${FRAME_SKIP:-4}"
export IMG_SIZE_0="${IMG_SIZE_0:-64}"
export IMG_SIZE_1="${IMG_SIZE_1:-64}"
export MODEL_SIZE="${MODEL_SIZE:-small}"
export WANDB_MODE="${WANDB_MODE:-online}"
export PROJECT_NAME="${PROJECT_NAME:-HTS-WM-Paper-Final-FlatMH}"
export RUN_GROUP="${RUN_GROUP:-atari26-flat-mh-$(date +%Y%m%d_%H%M%S)}"

GAMES=(
  Alien Amidar Assault Asterix BankHeist BattleZone Boxing Breakout
  ChopperCommand CrazyClimber DemonAttack Freeway Frostbite Gopher
  Hero Jamesbond Kangaroo Krull KungFuMaster MsPacman Pong PrivateEye
  Qbert RoadRunner Seaquest UpNDown
)

for GAME in "${GAMES[@]}"; do
  for SEED in 0 1 2 3 4; do
    CONFIG_FILE=examples/hierarchical_dreamer/config/generated_configs/flat_mh.yaml \
    RUN_NAME="flat-mh-${GAME}-rgb-small-seed${SEED}-100k" \
    ENV_ID="ALE/${GAME}-v5" \
    SEED="$SEED" \
    DEVICE="$DEVICE" \
    WANDB_MODE="$WANDB_MODE" \
    PROJECT_NAME="$PROJECT_NAME" \
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
    RUN_LOG_DIR="logs/training_scripts/${RUN_GROUP}" \
    examples/hierarchical_dreamer/train_ablation.sh
  done
done
BASH
```

## 4. Flat-SAE

```bash
bash <<'BASH'
set -euo pipefail
cd /mnt/disk1/backup_user/dat.tt2/xuance

export DEVICE="${DEVICE:-cuda:0}"
export OBS_TYPE="${OBS_TYPE:-rgb}"
export NUM_STACK="${NUM_STACK:-1}"
export FRAME_SKIP="${FRAME_SKIP:-4}"
export IMG_SIZE_0="${IMG_SIZE_0:-64}"
export IMG_SIZE_1="${IMG_SIZE_1:-64}"
export MODEL_SIZE="${MODEL_SIZE:-small}"
export WANDB_MODE="${WANDB_MODE:-online}"
export PROJECT_NAME="${PROJECT_NAME:-HTS-WM-Paper-Final-FlatSAE}"
export RUN_GROUP="${RUN_GROUP:-atari26-flat-sae-$(date +%Y%m%d_%H%M%S)}"

GAMES=(
  Alien Amidar Assault Asterix BankHeist BattleZone Boxing Breakout
  ChopperCommand CrazyClimber DemonAttack Freeway Frostbite Gopher
  Hero Jamesbond Kangaroo Krull KungFuMaster MsPacman Pong PrivateEye
  Qbert RoadRunner Seaquest UpNDown
)

for GAME in "${GAMES[@]}"; do
  for SEED in 0 1 2 3 4; do
    CONFIG_FILE=examples/hierarchical_dreamer/config/generated_configs/flat_sae.yaml \
    RUN_NAME="flat-sae-${GAME}-rgb-small-seed${SEED}-100k" \
    ENV_ID="ALE/${GAME}-v5" \
    SEED="$SEED" \
    DEVICE="$DEVICE" \
    WANDB_MODE="$WANDB_MODE" \
    PROJECT_NAME="$PROJECT_NAME" \
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
    RUN_LOG_DIR="logs/training_scripts/${RUN_GROUP}" \
    examples/hierarchical_dreamer/train_ablation.sh
  done
done
BASH
```

## 5. SGF-Style Flat Same-Code

```bash
bash <<'BASH'
set -euo pipefail
cd /mnt/disk1/backup_user/dat.tt2/xuance

export DEVICE="${DEVICE:-cuda:0}"
export OBS_TYPE="${OBS_TYPE:-rgb}"
export NUM_STACK="${NUM_STACK:-1}"
export FRAME_SKIP="${FRAME_SKIP:-4}"
export IMG_SIZE_0="${IMG_SIZE_0:-64}"
export IMG_SIZE_1="${IMG_SIZE_1:-64}"
export MODEL_SIZE="${MODEL_SIZE:-small}"
export WANDB_MODE="${WANDB_MODE:-online}"
export PROJECT_NAME="${PROJECT_NAME:-HTS-WM-Paper-Final-SGFStyleFlat}"
export RUN_GROUP="${RUN_GROUP:-atari26-sgf-style-flat-$(date +%Y%m%d_%H%M%S)}"

GAMES=(
  Alien Amidar Assault Asterix BankHeist BattleZone Boxing Breakout
  ChopperCommand CrazyClimber DemonAttack Freeway Frostbite Gopher
  Hero Jamesbond Kangaroo Krull KungFuMaster MsPacman Pong PrivateEye
  Qbert RoadRunner Seaquest UpNDown
)

for GAME in "${GAMES[@]}"; do
  for SEED in 0 1 2 3 4; do
    CONFIG_FILE=examples/hierarchical_dreamer/config/generated_configs/sgf_style_flat_same_code.yaml \
    RUN_NAME="sgf-style-flat-${GAME}-rgb-small-seed${SEED}-100k" \
    ENV_ID="ALE/${GAME}-v5" \
    SEED="$SEED" \
    DEVICE="$DEVICE" \
    WANDB_MODE="$WANDB_MODE" \
    PROJECT_NAME="$PROJECT_NAME" \
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
    RUN_LOG_DIR="logs/training_scripts/${RUN_GROUP}" \
    examples/hierarchical_dreamer/train_ablation.sh
  done
done
BASH
```

## 6. Recon-Only Hierarchy

```bash
bash <<'BASH'
set -euo pipefail
cd /mnt/disk1/backup_user/dat.tt2/xuance

export DEVICE="${DEVICE:-cuda:0}"
export OBS_TYPE="${OBS_TYPE:-rgb}"
export NUM_STACK="${NUM_STACK:-1}"
export FRAME_SKIP="${FRAME_SKIP:-4}"
export IMG_SIZE_0="${IMG_SIZE_0:-64}"
export IMG_SIZE_1="${IMG_SIZE_1:-64}"
export MODEL_SIZE="${MODEL_SIZE:-small}"
export WANDB_MODE="${WANDB_MODE:-online}"
export PROJECT_NAME="${PROJECT_NAME:-HTS-WM-Paper-Final-ReconOnly}"
export RUN_GROUP="${RUN_GROUP:-atari26-recon-only-$(date +%Y%m%d_%H%M%S)}"

GAMES=(
  Alien Amidar Assault Asterix BankHeist BattleZone Boxing Breakout
  ChopperCommand CrazyClimber DemonAttack Freeway Frostbite Gopher
  Hero Jamesbond Kangaroo Krull KungFuMaster MsPacman Pong PrivateEye
  Qbert RoadRunner Seaquest UpNDown
)

for GAME in "${GAMES[@]}"; do
  for SEED in 0 1 2 3 4; do
    CONFIG_FILE=examples/hierarchical_dreamer/config/generated_configs/recon_only_hierarchy.yaml \
    RUN_NAME="recon-only-hierarchy-${GAME}-rgb-small-seed${SEED}-100k" \
    ENV_ID="ALE/${GAME}-v5" \
    SEED="$SEED" \
    DEVICE="$DEVICE" \
    WANDB_MODE="$WANDB_MODE" \
    PROJECT_NAME="$PROJECT_NAME" \
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
    RUN_LOG_DIR="logs/training_scripts/${RUN_GROUP}" \
    examples/hierarchical_dreamer/train_ablation.sh
  done
done
BASH
```

## 7. Dense Multi-Stride No Sparse

```bash
bash <<'BASH'
set -euo pipefail
cd /mnt/disk1/backup_user/dat.tt2/xuance

export DEVICE="${DEVICE:-cuda:0}"
export OBS_TYPE="${OBS_TYPE:-rgb}"
export NUM_STACK="${NUM_STACK:-1}"
export FRAME_SKIP="${FRAME_SKIP:-4}"
export IMG_SIZE_0="${IMG_SIZE_0:-64}"
export IMG_SIZE_1="${IMG_SIZE_1:-64}"
export MODEL_SIZE="${MODEL_SIZE:-small}"
export WANDB_MODE="${WANDB_MODE:-online}"
export PROJECT_NAME="${PROJECT_NAME:-HTS-WM-Paper-Final-DenseMultiStrideNoSparse}"
export RUN_GROUP="${RUN_GROUP:-atari26-dense-multistride-no-sparse-$(date +%Y%m%d_%H%M%S)}"

GAMES=(
  Alien Amidar Assault Asterix BankHeist BattleZone Boxing Breakout
  ChopperCommand CrazyClimber DemonAttack Freeway Frostbite Gopher
  Hero Jamesbond Kangaroo Krull KungFuMaster MsPacman Pong PrivateEye
  Qbert RoadRunner Seaquest UpNDown
)

for GAME in "${GAMES[@]}"; do
  for SEED in 0 1 2 3 4; do
    CONFIG_FILE=examples/hierarchical_dreamer/config/generated_configs/dense_multistride_no_sparse.yaml \
    RUN_NAME="dense-multistride-no-sparse-${GAME}-rgb-small-seed${SEED}-100k" \
    ENV_ID="ALE/${GAME}-v5" \
    SEED="$SEED" \
    DEVICE="$DEVICE" \
    WANDB_MODE="$WANDB_MODE" \
    PROJECT_NAME="$PROJECT_NAME" \
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
    RUN_LOG_DIR="logs/training_scripts/${RUN_GROUP}" \
    examples/hierarchical_dreamer/train_ablation.sh
  done
done
BASH
```

## 8. HTS No Hierarchy

```bash
bash <<'BASH'
set -euo pipefail
cd /mnt/disk1/backup_user/dat.tt2/xuance

export DEVICE="${DEVICE:-cuda:0}"
export OBS_TYPE="${OBS_TYPE:-rgb}"
export NUM_STACK="${NUM_STACK:-1}"
export FRAME_SKIP="${FRAME_SKIP:-4}"
export IMG_SIZE_0="${IMG_SIZE_0:-64}"
export IMG_SIZE_1="${IMG_SIZE_1:-64}"
export MODEL_SIZE="${MODEL_SIZE:-small}"
export WANDB_MODE="${WANDB_MODE:-online}"
export PROJECT_NAME="${PROJECT_NAME:-HTS-WM-Paper-Final-NoHier}"
export RUN_GROUP="${RUN_GROUP:-atari26-hts-no-hier-$(date +%Y%m%d_%H%M%S)}"

GAMES=(
  Alien Amidar Assault Asterix BankHeist BattleZone Boxing Breakout
  ChopperCommand CrazyClimber DemonAttack Freeway Frostbite Gopher
  Hero Jamesbond Kangaroo Krull KungFuMaster MsPacman Pong PrivateEye
  Qbert RoadRunner Seaquest UpNDown
)

for GAME in "${GAMES[@]}"; do
  for SEED in 0 1 2 3 4; do
    CONFIG_FILE=examples/hierarchical_dreamer/config/generated_configs/hts_no_hier.yaml \
    RUN_NAME="hts-no-hier-${GAME}-rgb-small-seed${SEED}-100k" \
    ENV_ID="ALE/${GAME}-v5" \
    SEED="$SEED" \
    DEVICE="$DEVICE" \
    WANDB_MODE="$WANDB_MODE" \
    PROJECT_NAME="$PROJECT_NAME" \
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
    RUN_LOG_DIR="logs/training_scripts/${RUN_GROUP}" \
    examples/hierarchical_dreamer/train_ablation.sh
  done
done
BASH
```

## 9. HTS No Sparse Dynamics

```bash
bash <<'BASH'
set -euo pipefail
cd /mnt/disk1/backup_user/dat.tt2/xuance

export DEVICE="${DEVICE:-cuda:0}"
export OBS_TYPE="${OBS_TYPE:-rgb}"
export NUM_STACK="${NUM_STACK:-1}"
export FRAME_SKIP="${FRAME_SKIP:-4}"
export IMG_SIZE_0="${IMG_SIZE_0:-64}"
export IMG_SIZE_1="${IMG_SIZE_1:-64}"
export MODEL_SIZE="${MODEL_SIZE:-small}"
export WANDB_MODE="${WANDB_MODE:-online}"
export PROJECT_NAME="${PROJECT_NAME:-HTS-WM-Paper-Final-NoSDyn}"
export RUN_GROUP="${RUN_GROUP:-atari26-hts-no-sdyn-$(date +%Y%m%d_%H%M%S)}"

GAMES=(
  Alien Amidar Assault Asterix BankHeist BattleZone Boxing Breakout
  ChopperCommand CrazyClimber DemonAttack Freeway Frostbite Gopher
  Hero Jamesbond Kangaroo Krull KungFuMaster MsPacman Pong PrivateEye
  Qbert RoadRunner Seaquest UpNDown
)

for GAME in "${GAMES[@]}"; do
  for SEED in 0 1 2 3 4; do
    CONFIG_FILE=examples/hierarchical_dreamer/config/generated_configs/hts_no_sdyn.yaml \
    RUN_NAME="hts-no-sdyn-${GAME}-rgb-small-seed${SEED}-100k" \
    ENV_ID="ALE/${GAME}-v5" \
    SEED="$SEED" \
    DEVICE="$DEVICE" \
    WANDB_MODE="$WANDB_MODE" \
    PROJECT_NAME="$PROJECT_NAME" \
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
    RUN_LOG_DIR="logs/training_scripts/${RUN_GROUP}" \
    examples/hierarchical_dreamer/train_ablation.sh
  done
done
BASH
```

## 10. HTS No Temporal Consistency

```bash
bash <<'BASH'
set -euo pipefail
cd /mnt/disk1/backup_user/dat.tt2/xuance

export DEVICE="${DEVICE:-cuda:0}"
export OBS_TYPE="${OBS_TYPE:-rgb}"
export NUM_STACK="${NUM_STACK:-1}"
export FRAME_SKIP="${FRAME_SKIP:-4}"
export IMG_SIZE_0="${IMG_SIZE_0:-64}"
export IMG_SIZE_1="${IMG_SIZE_1:-64}"
export MODEL_SIZE="${MODEL_SIZE:-small}"
export WANDB_MODE="${WANDB_MODE:-online}"
export PROJECT_NAME="${PROJECT_NAME:-HTS-WM-Paper-Final-NoTemp}"
export RUN_GROUP="${RUN_GROUP:-atari26-hts-no-temp-$(date +%Y%m%d_%H%M%S)}"

GAMES=(
  Alien Amidar Assault Asterix BankHeist BattleZone Boxing Breakout
  ChopperCommand CrazyClimber DemonAttack Freeway Frostbite Gopher
  Hero Jamesbond Kangaroo Krull KungFuMaster MsPacman Pong PrivateEye
  Qbert RoadRunner Seaquest UpNDown
)

for GAME in "${GAMES[@]}"; do
  for SEED in 0 1 2 3 4; do
    CONFIG_FILE=examples/hierarchical_dreamer/config/generated_configs/hts_no_temp.yaml \
    RUN_NAME="hts-no-temp-${GAME}-rgb-small-seed${SEED}-100k" \
    ENV_ID="ALE/${GAME}-v5" \
    SEED="$SEED" \
    DEVICE="$DEVICE" \
    WANDB_MODE="$WANDB_MODE" \
    PROJECT_NAME="$PROJECT_NAME" \
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
    RUN_LOG_DIR="logs/training_scripts/${RUN_GROUP}" \
    examples/hierarchical_dreamer/train_ablation.sh
  done
done
BASH
```

## 11. HTS No Variance/Covariance

```bash
bash <<'BASH'
set -euo pipefail
cd /mnt/disk1/backup_user/dat.tt2/xuance

export DEVICE="${DEVICE:-cuda:0}"
export OBS_TYPE="${OBS_TYPE:-rgb}"
export NUM_STACK="${NUM_STACK:-1}"
export FRAME_SKIP="${FRAME_SKIP:-4}"
export IMG_SIZE_0="${IMG_SIZE_0:-64}"
export IMG_SIZE_1="${IMG_SIZE_1:-64}"
export MODEL_SIZE="${MODEL_SIZE:-small}"
export WANDB_MODE="${WANDB_MODE:-online}"
export PROJECT_NAME="${PROJECT_NAME:-HTS-WM-Paper-Final-NoVC}"
export RUN_GROUP="${RUN_GROUP:-atari26-hts-no-vc-$(date +%Y%m%d_%H%M%S)}"

GAMES=(
  Alien Amidar Assault Asterix BankHeist BattleZone Boxing Breakout
  ChopperCommand CrazyClimber DemonAttack Freeway Frostbite Gopher
  Hero Jamesbond Kangaroo Krull KungFuMaster MsPacman Pong PrivateEye
  Qbert RoadRunner Seaquest UpNDown
)

for GAME in "${GAMES[@]}"; do
  for SEED in 0 1 2 3 4; do
    CONFIG_FILE=examples/hierarchical_dreamer/config/generated_configs/hts_no_vc.yaml \
    RUN_NAME="hts-no-vc-${GAME}-rgb-small-seed${SEED}-100k" \
    ENV_ID="ALE/${GAME}-v5" \
    SEED="$SEED" \
    DEVICE="$DEVICE" \
    WANDB_MODE="$WANDB_MODE" \
    PROJECT_NAME="$PROJECT_NAME" \
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
    RUN_LOG_DIR="logs/training_scripts/${RUN_GROUP}" \
    examples/hierarchical_dreamer/train_ablation.sh
  done
done
BASH
```

## 12. XuanCe HarmonyDream

```bash
bash <<'BASH'
set -euo pipefail
cd /mnt/disk1/backup_user/dat.tt2/xuance

export DEVICE="${DEVICE:-cuda:0}"
export OBS_TYPE="${OBS_TYPE:-rgb}"
export NUM_STACK="${NUM_STACK:-1}"
export FRAME_SKIP="${FRAME_SKIP:-4}"
export IMG_SIZE_0="${IMG_SIZE_0:-64}"
export IMG_SIZE_1="${IMG_SIZE_1:-64}"
export MODEL_SIZE="${MODEL_SIZE:-small}"
export WANDB_MODE="${WANDB_MODE:-online}"
export PROJECT_NAME="${PROJECT_NAME:-HTS-WM-Paper-Final-XuanCeHarmony}"
export RUN_GROUP="${RUN_GROUP:-atari26-xuance-harmony-$(date +%Y%m%d_%H%M%S)}"

GAMES=(
  Alien Amidar Assault Asterix BankHeist BattleZone Boxing Breakout
  ChopperCommand CrazyClimber DemonAttack Freeway Frostbite Gopher
  Hero Jamesbond Kangaroo Krull KungFuMaster MsPacman Pong PrivateEye
  Qbert RoadRunner Seaquest UpNDown
)

for GAME in "${GAMES[@]}"; do
  ENV_ID="ALE/${GAME}-v5" \
  METHODS=harmony \
  SEEDS=0,1,2,3,4 \
  DEVICE="$DEVICE" \
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

## 13. TSAE-Style

```bash
bash <<'BASH'
set -euo pipefail
cd /mnt/disk1/backup_user/dat.tt2/xuance

export DEVICE="${DEVICE:-cuda:0}"
export OBS_TYPE="${OBS_TYPE:-rgb}"
export NUM_STACK="${NUM_STACK:-1}"
export FRAME_SKIP="${FRAME_SKIP:-4}"
export IMG_SIZE_0="${IMG_SIZE_0:-64}"
export IMG_SIZE_1="${IMG_SIZE_1:-64}"
export MODEL_SIZE="${MODEL_SIZE:-small}"
export WANDB_MODE="${WANDB_MODE:-online}"
export PROJECT_NAME="${PROJECT_NAME:-HTS-WM-Paper-Final-TSAEStyle}"
export RUN_GROUP="${RUN_GROUP:-atari26-tsae-style-$(date +%Y%m%d_%H%M%S)}"

GAMES=(
  Alien Amidar Assault Asterix BankHeist BattleZone Boxing Breakout
  ChopperCommand CrazyClimber DemonAttack Freeway Frostbite Gopher
  Hero Jamesbond Kangaroo Krull KungFuMaster MsPacman Pong PrivateEye
  Qbert RoadRunner Seaquest UpNDown
)

for GAME in "${GAMES[@]}"; do
  ENV_ID="ALE/${GAME}-v5" \
  METHODS=tsae \
  SEEDS=0,1,2,3,4 \
  DEVICE="$DEVICE" \
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

## Optional: Disable Eval Videos

Video logging is useful for visual inspection but can use W&B storage. To run
without final evaluation videos, replace this line inside the block:

```bash
RENDER_EVAL_VIDEO=true \
```

with:

```bash
RENDER_EVAL_VIDEO=false \
```

## Check Failures

Run this after a queue finishes:

```bash
cd /mnt/disk1/backup_user/dat.tt2/xuance

rg -n "Traceback|RuntimeError|Exception|CUDA out|nan|NaN" logs/training_scripts -S
```

## Aggregate Training Return Curves

This builds Dreamer-style training episode return curves from
`train_episode_returns.csv` files:

```bash
cd /mnt/disk1/backup_user/dat.tt2/xuance

.venv/bin/python examples/hierarchical_dreamer/scripts/aggregate_train_episode_curves.py \
  --input-root logs/paper-final \
  --output artifacts/curves/paper_final_train_episode_return_curves.csv \
  --bins 30 \
  --max-env-frames 400000
```

## Result Files To Send Back

After each completed method, send:

```text
W&B project name
run group
failed runs, if any
logs/training_scripts/<the_run_group_printed_by_the_command>/*.log
artifacts/paper_development/atari100k/*/raw_metrics.json, if present
logs/paper-final/**/final_eval.json
logs/paper-final/**/train_episode_returns.csv
```
