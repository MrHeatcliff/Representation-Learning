#!/usr/bin/env bash
set -euo pipefail

cd /mnt/disk1/backup_user/dat.tt2/xuance

export PYTHON_BIN="${PYTHON_BIN:-/mnt/disk1/backup_user/dat.tt2/xuance/.venv/bin/python}"
export ENV_ID="${ENV_ID:-ALE/Breakout-v5}"
export DEVICE="${DEVICE:-cuda:0}"
export SEED="${SEED:-1}"
export WANDB_MODE="${WANDB_MODE:-online}"
export PROJECT_NAME="${PROJECT_NAME:-HTS-WM-P0-Smoke}"
export RUNNING_STEPS="${RUNNING_STEPS:-10000}"
export EVAL_INTERVAL="${EVAL_INTERVAL:-2500}"
export REPLAY_RATIO="${REPLAY_RATIO:-1.0}"
export BUFFER_SIZE="${BUFFER_SIZE:-1000000}"
export START_TRAINING="${START_TRAINING:-1024}"
export BATCH_SIZE="${BATCH_SIZE:-16}"
export SEQ_LEN="${SEQ_LEN:-64}"
export PARALLELS="${PARALLELS:-1}"
export BENCHMARK="${BENCHMARK:-1}"
export CHECKPOINT_RULE="${CHECKPOINT_RULE:-final}"
export EVAL_PROTOCOL="${EVAL_PROTOCOL:-final}"
export TEST_EPISODE="${TEST_EPISODE:-3}"
export INTERMEDIATE_TEST_EPISODE="${INTERMEDIATE_TEST_EPISODE:-3}"
export RENDER_EVAL_VIDEO="${RENDER_EVAL_VIDEO:-true}"
export RENDER_INTERMEDIATE_VIDEO="${RENDER_INTERMEDIATE_VIDEO:-false}"
export OBS_TYPE="${OBS_TYPE:-rgb}"
export NUM_STACK="${NUM_STACK:-1}"
export FRAME_SKIP="${FRAME_SKIP:-4}"
export REPEAT_ACTION_PROBABILITY="${REPEAT_ACTION_PROBABILITY:-0.0}"
export CLIP_REWARD="${CLIP_REWARD:-false}"
export EPISODIC_LIFE="${EPISODIC_LIFE:-false}"
export IMG_SIZE_0="${IMG_SIZE_0:-64}"
export IMG_SIZE_1="${IMG_SIZE_1:-64}"
export MODEL_SIZE="${MODEL_SIZE:-small}"
export SMOKE_TAG="${SMOKE_TAG:-p0-smoke-10k}"

run_dreamer() {
  local run_name="smoke-dreamerv3-breakout-seed${SEED}-${RUNNING_STEPS}steps-${SMOKE_TAG}"
  echo "===== RUN ${run_name} ====="
  "${PYTHON_BIN}" examples/dreamer_v3/dreamer_v3_atari.py \
    --env-id "${ENV_ID}" \
    --obs-type "${OBS_TYPE}" \
    --num-stack "${NUM_STACK}" \
    --frame-skip "${FRAME_SKIP}" \
    --repeat-action-probability "${REPEAT_ACTION_PROBABILITY}" \
    --clip-reward "${CLIP_REWARD}" \
    --episodic-life "${EPISODIC_LIFE}" \
    --img-size "${IMG_SIZE_0}" "${IMG_SIZE_1}" \
    --model-size "${MODEL_SIZE}" \
    --device "${DEVICE}" \
    --seed "${SEED}" \
    --logger wandb \
    --project-name "${PROJECT_NAME}" \
    --wandb-mode "${WANDB_MODE}" \
    --wandb-run-name "${run_name}" \
    --running-steps "${RUNNING_STEPS}" \
    --eval-interval "${EVAL_INTERVAL}" \
    --replay-ratio "${REPLAY_RATIO}" \
    --buffer-size "${BUFFER_SIZE}" \
    --start-training "${START_TRAINING}" \
    --batch-size "${BATCH_SIZE}" \
    --seq-len "${SEQ_LEN}" \
    --parallels "${PARALLELS}" \
    --benchmark "${BENCHMARK}" \
    --test-episode "${TEST_EPISODE}" \
    --intermediate-test-episode "${INTERMEDIATE_TEST_EPISODE}" \
    --render-eval-video "${RENDER_EVAL_VIDEO}" \
    --render-intermediate-video "${RENDER_INTERMEDIATE_VIDEO}" \
    --checkpoint-rule "${CHECKPOINT_RULE}" \
    --eval-protocol "${EVAL_PROTOCOL}" \
    --log-dir "logs/p0_smoke/${run_name}/" \
    --model-dir "models/p0_smoke/${run_name}/"
}

run_hts_config() {
  local condition="$1"
  local config_file="$2"
  local run_name="smoke-${condition}-breakout-seed${SEED}-${RUNNING_STEPS}steps-${SMOKE_TAG}"
  echo "===== RUN ${run_name} ====="
  CONFIG_FILE="${config_file}" \
  RUN_NAME="${run_name}" \
  ENV_ID="${ENV_ID}" \
  OBS_TYPE="${OBS_TYPE}" \
  NUM_STACK="${NUM_STACK}" \
  FRAME_SKIP="${FRAME_SKIP}" \
  REPEAT_ACTION_PROBABILITY="${REPEAT_ACTION_PROBABILITY}" \
  CLIP_REWARD="${CLIP_REWARD}" \
  EPISODIC_LIFE="${EPISODIC_LIFE}" \
  IMG_SIZE_0="${IMG_SIZE_0}" \
  IMG_SIZE_1="${IMG_SIZE_1}" \
  MODEL_SIZE="${MODEL_SIZE}" \
  DEVICE="${DEVICE}" \
  SEED="${SEED}" \
  WANDB_MODE="${WANDB_MODE}" \
  PROJECT_NAME="${PROJECT_NAME}" \
  RUNNING_STEPS="${RUNNING_STEPS}" \
  EVAL_INTERVAL="${EVAL_INTERVAL}" \
  REPLAY_RATIO="${REPLAY_RATIO}" \
  BUFFER_SIZE="${BUFFER_SIZE}" \
  START_TRAINING="${START_TRAINING}" \
  BATCH_SIZE="${BATCH_SIZE}" \
  SEQ_LEN="${SEQ_LEN}" \
  PARALLELS="${PARALLELS}" \
  BENCHMARK="${BENCHMARK}" \
  CHECKPOINT_RULE="${CHECKPOINT_RULE}" \
  EVAL_PROTOCOL="${EVAL_PROTOCOL}" \
  TEST_EPISODE="${TEST_EPISODE}" \
  INTERMEDIATE_TEST_EPISODE="${INTERMEDIATE_TEST_EPISODE}" \
  RENDER_EVAL_VIDEO="${RENDER_EVAL_VIDEO}" \
  RENDER_INTERMEDIATE_VIDEO="${RENDER_INTERMEDIATE_VIDEO}" \
  LOG_DIR="logs/p0_smoke/${run_name}/" \
  MODEL_DIR="models/p0_smoke/${run_name}/" \
  examples/hierarchical_dreamer/train_ablation.sh
}

run_xuance_harmony() {
  local run_name="smoke-xuance-harmonydream-breakout-seed${SEED}-${RUNNING_STEPS}steps-${SMOKE_TAG}"
  echo "===== RUN ${run_name} ====="
  RUN_NAME="${run_name}" \
  ENV_ID="${ENV_ID}" \
  OBS_TYPE="${OBS_TYPE}" \
  NUM_STACK="${NUM_STACK}" \
  FRAME_SKIP="${FRAME_SKIP}" \
  REPEAT_ACTION_PROBABILITY="${REPEAT_ACTION_PROBABILITY}" \
  CLIP_REWARD="${CLIP_REWARD}" \
  EPISODIC_LIFE="${EPISODIC_LIFE}" \
  IMG_SIZE_0="${IMG_SIZE_0}" \
  IMG_SIZE_1="${IMG_SIZE_1}" \
  MODEL_SIZE="${MODEL_SIZE}" \
  DEVICE="${DEVICE}" \
  SEED="${SEED}" \
  WANDB_MODE="${WANDB_MODE}" \
  PROJECT_NAME="${PROJECT_NAME}" \
  RUNNING_STEPS="${RUNNING_STEPS}" \
  EVAL_INTERVAL="${EVAL_INTERVAL}" \
  REPLAY_RATIO="${REPLAY_RATIO}" \
  BUFFER_SIZE="${BUFFER_SIZE}" \
  START_TRAINING="${START_TRAINING}" \
  BATCH_SIZE="${BATCH_SIZE}" \
  SEQ_LEN="${SEQ_LEN}" \
  PARALLELS="${PARALLELS}" \
  BENCHMARK="${BENCHMARK}" \
  CHECKPOINT_RULE="${CHECKPOINT_RULE}" \
  EVAL_PROTOCOL="${EVAL_PROTOCOL}" \
  TEST_EPISODE="${TEST_EPISODE}" \
  RENDER_EVAL_VIDEO="${RENDER_EVAL_VIDEO}" \
  RENDER_INTERMEDIATE_VIDEO="${RENDER_INTERMEDIATE_VIDEO}" \
  LOG_DIR="logs/p0_smoke/${run_name}/" \
  MODEL_DIR="models/p0_smoke/${run_name}/" \
  examples/hierarchical_dreamer/baselines/run_xuance_harmonydream_atari100k.sh
}

run_tsae_style() {
  local run_name="smoke-tsae-style-breakout-seed${SEED}-${RUNNING_STEPS}steps-${SMOKE_TAG}"
  echo "===== RUN ${run_name} ====="
  RUN_NAME="${run_name}" \
  ENV_ID="${ENV_ID}" \
  OBS_TYPE="${OBS_TYPE}" \
  NUM_STACK="${NUM_STACK}" \
  FRAME_SKIP="${FRAME_SKIP}" \
  REPEAT_ACTION_PROBABILITY="${REPEAT_ACTION_PROBABILITY}" \
  CLIP_REWARD="${CLIP_REWARD}" \
  EPISODIC_LIFE="${EPISODIC_LIFE}" \
  IMG_SIZE_0="${IMG_SIZE_0}" \
  IMG_SIZE_1="${IMG_SIZE_1}" \
  MODEL_SIZE="${MODEL_SIZE}" \
  DEVICE="${DEVICE}" \
  SEED="${SEED}" \
  WANDB_MODE="${WANDB_MODE}" \
  PROJECT_NAME="${PROJECT_NAME}" \
  RUNNING_STEPS="${RUNNING_STEPS}" \
  EVAL_INTERVAL="${EVAL_INTERVAL}" \
  REPLAY_RATIO="${REPLAY_RATIO}" \
  BUFFER_SIZE="${BUFFER_SIZE}" \
  START_TRAINING="${START_TRAINING}" \
  BATCH_SIZE="${BATCH_SIZE}" \
  SEQ_LEN="${SEQ_LEN}" \
  PARALLELS="${PARALLELS}" \
  BENCHMARK="${BENCHMARK}" \
  CHECKPOINT_RULE="${CHECKPOINT_RULE}" \
  EVAL_PROTOCOL="${EVAL_PROTOCOL}" \
  TEST_EPISODE="${TEST_EPISODE}" \
  RENDER_EVAL_VIDEO="${RENDER_EVAL_VIDEO}" \
  RENDER_INTERMEDIATE_VIDEO="${RENDER_INTERMEDIATE_VIDEO}" \
  LOG_DIR="logs/p0_smoke/${run_name}/" \
  MODEL_DIR="models/p0_smoke/${run_name}/" \
  examples/hierarchical_dreamer/baselines/run_tsae_style_atari100k.sh
}

# run_dreamer
# run_hts_config "hts-full" "config/atari100k_two_phase.yaml"
# run_hts_config "flat-sae" "config/generated_configs/flat_sae.yaml"
# run_hts_config "flat-mh" "config/generated_configs/flat_mh.yaml"
# run_hts_config "sgf-style-flat-same-code" "config/generated_configs/sgf_style_flat_same_code.yaml"
# run_hts_config "recon-only-hierarchy" "config/generated_configs/recon_only_hierarchy.yaml"
# run_hts_config "dense-multistride-no-sparse" "config/generated_configs/dense_multistride_no_sparse.yaml"
# run_hts_config "hts-no-hier" "config/generated_configs/hts_no_hier.yaml"
run_hts_config "hts-no-sdyn" "config/generated_configs/hts_no_sdyn.yaml"
run_hts_config "hts-no-temp" "config/generated_configs/hts_no_temp.yaml"
run_hts_config "hts-no-vc" "config/generated_configs/hts_no_vc.yaml"
run_hts_config "larger-flat-param" "config/generated_configs/larger_flat_param_breakout.yaml"
run_xuance_harmony
run_tsae_style

echo "All P0 Breakout smoke runs finished."
