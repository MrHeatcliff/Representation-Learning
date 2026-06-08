#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"

METHODS="${METHODS:-dreamer,htswm}"
SEEDS="${SEEDS:-0,1,2,3,4}"
ENV_ID="${ENV_ID:-ALE/Breakout-v5}"
ROM_NAME="${ENV_ID##*/}"
DEVICE="${DEVICE:-cuda:0}"
WANDB_MODE="${WANDB_MODE:-online}"
PROJECT_NAME="${PROJECT_NAME:-HTS-WM-Paper-Final}"
RUNNING_STEPS="${RUNNING_STEPS:-100000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-2000}"
REPLAY_RATIO="${REPLAY_RATIO:-0.125}"
BUFFER_SIZE="${BUFFER_SIZE:-1000000}"
START_TRAINING="${START_TRAINING:-1024}"
BATCH_SIZE="${BATCH_SIZE:-16}"
SEQ_LEN="${SEQ_LEN:-64}"
PARALLELS="${PARALLELS:-1}"
BENCHMARK="${BENCHMARK:-1}"
TEST_EPISODE="${TEST_EPISODE:-100}"
INTERMEDIATE_TEST_EPISODE="${INTERMEDIATE_TEST_EPISODE:-20}"
CHECKPOINT_RULE="${CHECKPOINT_RULE:-final}"
EVAL_PROTOCOL="${EVAL_PROTOCOL:-final}"
RENDER_EVAL_VIDEO="${RENDER_EVAL_VIDEO:-true}"
RENDER_INTERMEDIATE_VIDEO="${RENDER_INTERMEDIATE_VIDEO:-false}"
RUN_GROUP="${RUN_GROUP:-paper_final_atari100k_$(date +%Y%m%d_%H%M%S)}"
RUN_LOG_ROOT="${RUN_LOG_ROOT:-${REPO_ROOT}/logs/training_scripts/${RUN_GROUP}}"

mkdir -p "${RUN_LOG_ROOT}"

contains_method() {
  local needle="$1"
  [[ ",${METHODS}," == *",${needle},"* || "${METHODS}" == "all" ]]
}

run_dreamer() {
  local seed="$1"
  local run_name="DreamerV3-paper-final-${ROM_NAME}-seed${seed}-${RUNNING_STEPS}steps"
  local log_file="${RUN_LOG_ROOT}/${run_name}.log"
  (
    cd "${REPO_ROOT}"
    "${PYTHON_BIN}" examples/dreamer_v3/dreamer_v3_atari.py \
      --env-id "${ENV_ID}" \
      --device "${DEVICE}" \
      --seed "${seed}" \
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
      --log-dir "logs/paper-final/dreamer-v3/${ROM_NAME}/seed${seed}/" \
      --model-dir "models/paper-final/dreamer-v3/${ROM_NAME}/seed${seed}/"
  ) 2>&1 | tee "${log_file}"
}

run_htswm() {
  local seed="$1"
  local run_name="HTS-WM-paper-final-${ROM_NAME}-seed${seed}-${RUNNING_STEPS}steps"
  CONFIG_FILE=config/atari100k_two_phase.yaml \
  RUN_NAME="${run_name}" \
  ENV_ID="${ENV_ID}" \
  SEED="${seed}" \
  DEVICE="${DEVICE}" \
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
  TEST_EPISODE="${TEST_EPISODE}" \
  INTERMEDIATE_TEST_EPISODE="${INTERMEDIATE_TEST_EPISODE}" \
  RENDER_EVAL_VIDEO="${RENDER_EVAL_VIDEO}" \
  RENDER_INTERMEDIATE_VIDEO="${RENDER_INTERMEDIATE_VIDEO}" \
  CHECKPOINT_RULE="${CHECKPOINT_RULE}" \
  EVAL_PROTOCOL="${EVAL_PROTOCOL}" \
  PYTHON_BIN="${PYTHON_BIN}" \
  LOG_DIR="logs/paper-final/hts-wm/${ROM_NAME}/seed${seed}/" \
  MODEL_DIR="models/paper-final/hts-wm/${ROM_NAME}/seed${seed}/" \
  RUN_LOG_DIR="${RUN_LOG_ROOT}" \
  "${REPO_ROOT}/examples/hierarchical_dreamer/train_ablation.sh"
}

run_harmony() {
  local seed="$1"
  local run_name="XuanCe-HarmonyDream-paper-final-${ROM_NAME}-seed${seed}-${RUNNING_STEPS}steps"
  RUN_NAME="${run_name}" \
  ENV_ID="${ENV_ID}" \
  SEED="${seed}" \
  DEVICE="${DEVICE}" \
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
  TEST_EPISODE="${TEST_EPISODE}" \
  INTERMEDIATE_TEST_EPISODE="${INTERMEDIATE_TEST_EPISODE}" \
  RENDER_EVAL_VIDEO="${RENDER_EVAL_VIDEO}" \
  RENDER_INTERMEDIATE_VIDEO="${RENDER_INTERMEDIATE_VIDEO}" \
  CHECKPOINT_RULE="${CHECKPOINT_RULE}" \
  EVAL_PROTOCOL="${EVAL_PROTOCOL}" \
  PYTHON_BIN="${PYTHON_BIN}" \
  LOG_DIR="logs/paper-final/harmonydream/${ROM_NAME}/seed${seed}/" \
  MODEL_DIR="models/paper-final/harmonydream/${ROM_NAME}/seed${seed}/" \
  RUN_LOG_DIR="${RUN_LOG_ROOT}" \
  "${REPO_ROOT}/examples/hierarchical_dreamer/baselines/run_xuance_harmonydream_atari100k.sh"
}

run_tsae() {
  local seed="$1"
  local run_name="TSAE-style-paper-final-${ROM_NAME}-seed${seed}-${RUNNING_STEPS}steps"
  RUN_NAME="${run_name}" \
  ENV_ID="${ENV_ID}" \
  SEED="${seed}" \
  DEVICE="${DEVICE}" \
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
  TEST_EPISODE="${TEST_EPISODE}" \
  INTERMEDIATE_TEST_EPISODE="${INTERMEDIATE_TEST_EPISODE}" \
  RENDER_EVAL_VIDEO="${RENDER_EVAL_VIDEO}" \
  RENDER_INTERMEDIATE_VIDEO="${RENDER_INTERMEDIATE_VIDEO}" \
  CHECKPOINT_RULE="${CHECKPOINT_RULE}" \
  EVAL_PROTOCOL="${EVAL_PROTOCOL}" \
  PYTHON_BIN="${PYTHON_BIN}" \
  LOG_DIR="logs/paper-final/tsae-style/${ROM_NAME}/seed${seed}/" \
  MODEL_DIR="models/paper-final/tsae-style/${ROM_NAME}/seed${seed}/" \
  RUN_LOG_DIR="${RUN_LOG_ROOT}" \
  "${REPO_ROOT}/examples/hierarchical_dreamer/baselines/run_tsae_style_atari100k.sh"
}

echo "Run group: ${RUN_GROUP}"
echo "Methods: ${METHODS}"
echo "Seeds: ${SEEDS}"
echo "Environment: ${ENV_ID}"
echo "Checkpoint rule: ${CHECKPOINT_RULE}"
echo "Test episodes: ${TEST_EPISODE}"
echo "Intermediate test episodes: ${INTERMEDIATE_TEST_EPISODE}"
echo "Logs: ${RUN_LOG_ROOT}"

IFS=',' read -ra seed_array <<< "${SEEDS}"
for seed in "${seed_array[@]}"; do
  if contains_method dreamer; then
    run_dreamer "${seed}"
  fi
  if contains_method htswm; then
    run_htswm "${seed}"
  fi
  if contains_method harmony; then
    run_harmony "${seed}"
  fi
  if contains_method tsae; then
    run_tsae "${seed}"
  fi
done

echo "Paper-final same-code Atari100K queue finished."
