#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"

ENV_ID="${ENV_ID:-ALE/Breakout-v5}"
ROM_NAME="${ENV_ID##*/}"
DEVICE="${DEVICE:-cuda:0}"
WANDB_MODE="${WANDB_MODE:-online}"
PROJECT_NAME="${PROJECT_NAME:-Hierarchical-Dreamer-Atari100K}"

RUNNING_STEPS="${RUNNING_STEPS:-100000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-2000}"
REPLAY_RATIO="${REPLAY_RATIO:-1}"
BUFFER_SIZE="${BUFFER_SIZE:-1000000}"
START_TRAINING="${START_TRAINING:-1024}"
BATCH_SIZE="${BATCH_SIZE:-16}"
SEQ_LEN="${SEQ_LEN:-64}"
PARALLELS="${PARALLELS:-1}"
BENCHMARK="${BENCHMARK:-1}"

RUN_GROUP="${RUN_GROUP:-atari100k_$(date +%Y%m%d_%H%M%S)}"
RUN_LOG_ROOT="${RUN_LOG_ROOT:-${REPO_ROOT}/logs/training_scripts/${RUN_GROUP}}"
mkdir -p "${RUN_LOG_ROOT}"

run_hierarchical_regime() {
  local regime="$1"
  local config_file="$2"
  local log_file="${RUN_LOG_ROOT}/${regime}.log"

  mkdir -p \
    "${REPO_ROOT}/logs/hierarchical-dreamer/atari100k/${regime}/${ROM_NAME}" \
    "${REPO_ROOT}/models/hierarchical-dreamer/atari100k/${regime}/${ROM_NAME}"

  echo "[$(date --iso-8601=seconds)] Starting hierarchical regime: ${regime}" | tee "${log_file}"
  (
    cd "${REPO_ROOT}"
    "${PYTHON_BIN}" examples/hierarchical_dreamer/hierarchical_dreamer_atari.py \
      --config-file "${config_file}" \
      --env-id "${ENV_ID}" \
      --device "${DEVICE}" \
      --logger wandb \
      --project-name "${PROJECT_NAME}" \
      --wandb-mode "${WANDB_MODE}" \
      --running-steps "${RUNNING_STEPS}" \
      --eval-interval "${EVAL_INTERVAL}" \
      --replay-ratio "${REPLAY_RATIO}" \
      --buffer-size "${BUFFER_SIZE}" \
      --start-training "${START_TRAINING}" \
      --batch-size "${BATCH_SIZE}" \
      --seq-len "${SEQ_LEN}" \
      --parallels "${PARALLELS}" \
      --benchmark "${BENCHMARK}" \
      --log-dir "logs/hierarchical-dreamer/atari100k/${regime}/${ROM_NAME}/" \
      --model-dir "models/hierarchical-dreamer/atari100k/${regime}/${ROM_NAME}/"
  ) 2>&1 | tee -a "${log_file}"
  echo "[$(date --iso-8601=seconds)] Finished hierarchical regime: ${regime}" | tee -a "${log_file}"
}

run_dreamer_baseline() {
  local regime="dreamer_baseline"
  local log_file="${RUN_LOG_ROOT}/${regime}.log"

  mkdir -p \
    "${REPO_ROOT}/logs/dreamer-v3/atari100k/${regime}/${ROM_NAME}" \
    "${REPO_ROOT}/models/dreamer-v3/atari100k/${regime}/${ROM_NAME}"

  echo "[$(date --iso-8601=seconds)] Starting baseline: DreamerV3" | tee "${log_file}"
  (
    cd "${REPO_ROOT}"
    "${PYTHON_BIN}" examples/dreamer_v3/dreamer_v3_atari.py \
      --env-id "${ENV_ID}" \
      --device "${DEVICE}" \
      --logger wandb \
      --project-name "${PROJECT_NAME}" \
      --wandb-mode "${WANDB_MODE}" \
      --running-steps "${RUNNING_STEPS}" \
      --eval-interval "${EVAL_INTERVAL}" \
      --replay-ratio "${REPLAY_RATIO}" \
      --buffer-size "${BUFFER_SIZE}" \
      --start-training "${START_TRAINING}" \
      --batch-size "${BATCH_SIZE}" \
      --seq-len "${SEQ_LEN}" \
      --parallels "${PARALLELS}" \
      --benchmark "${BENCHMARK}" \
      --log-dir "logs/dreamer-v3/atari100k/${regime}/${ROM_NAME}/" \
      --model-dir "models/dreamer-v3/atari100k/${regime}/${ROM_NAME}/"
  ) 2>&1 | tee -a "${log_file}"
  echo "[$(date --iso-8601=seconds)] Finished baseline: DreamerV3" | tee -a "${log_file}"
}

echo "Run group: ${RUN_GROUP}"
echo "Run logs: ${RUN_LOG_ROOT}"
echo "Environment: ${ENV_ID}"
echo "Device: ${DEVICE}"
echo "W&B mode: ${WANDB_MODE}"

run_hierarchical_regime "frozen_encoder" "config/atari100k_frozen_encoder.yaml"
run_hierarchical_regime "two_phase" "config/atari100k_two_phase.yaml"
run_hierarchical_regime "fully_joint" "config/atari100k_fully_joint.yaml"
run_dreamer_baseline

echo "All Atari100K runs finished. Logs are in ${RUN_LOG_ROOT}"
