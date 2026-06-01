#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"

CONFIG_FILE="${CONFIG_FILE:?Set CONFIG_FILE to a YAML config path.}"
RUN_NAME="${RUN_NAME:?Set RUN_NAME to a descriptive W&B run name.}"
ENV_ID="${ENV_ID:-ALE/Breakout-v5}"
ROM_NAME="${ENV_ID##*/}"
DEVICE="${DEVICE:-cuda:0}"
WANDB_MODE="${WANDB_MODE:-online}"
PROJECT_NAME="${PROJECT_NAME:-HTS-WM-Ablations}"
RUNNING_STEPS="${RUNNING_STEPS:-100000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-2000}"
REPLAY_RATIO="${REPLAY_RATIO:-0.25}"
BUFFER_SIZE="${BUFFER_SIZE:-1000000}"
START_TRAINING="${START_TRAINING:-1024}"
BATCH_SIZE="${BATCH_SIZE:-16}"
SEQ_LEN="${SEQ_LEN:-64}"
PARALLELS="${PARALLELS:-1}"
BENCHMARK="${BENCHMARK:-1}"

LOG_DIR="${LOG_DIR:-logs/hierarchical-dreamer/ablations/${RUN_NAME}/${ROM_NAME}/}"
MODEL_DIR="${MODEL_DIR:-models/hierarchical-dreamer/ablations/${RUN_NAME}/${ROM_NAME}/}"
RUN_LOG_DIR="${RUN_LOG_DIR:-${REPO_ROOT}/logs/training_scripts/ablations}"
mkdir -p "${RUN_LOG_DIR}" "${REPO_ROOT}/${LOG_DIR}" "${REPO_ROOT}/${MODEL_DIR}"

(
  cd "${REPO_ROOT}"
  "${PYTHON_BIN}" examples/hierarchical_dreamer/hierarchical_dreamer_atari.py \
    --config-file "${CONFIG_FILE}" \
    --env-id "${ENV_ID}" \
    --device "${DEVICE}" \
    --logger wandb \
    --project-name "${PROJECT_NAME}" \
    --wandb-mode "${WANDB_MODE}" \
    --wandb-run-name "${RUN_NAME}" \
    --running-steps "${RUNNING_STEPS}" \
    --eval-interval "${EVAL_INTERVAL}" \
    --replay-ratio "${REPLAY_RATIO}" \
    --buffer-size "${BUFFER_SIZE}" \
    --start-training "${START_TRAINING}" \
    --batch-size "${BATCH_SIZE}" \
    --seq-len "${SEQ_LEN}" \
    --parallels "${PARALLELS}" \
    --benchmark "${BENCHMARK}" \
    --log-dir "${LOG_DIR}" \
    --model-dir "${MODEL_DIR}"
) 2>&1 | tee "${RUN_LOG_DIR}/${RUN_NAME}.log"
