#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"

ENV_ID="${ENV_ID:-ALE/Breakout-v5}"
ROM_NAME="${ENV_ID##*/}"
DEVICE="${DEVICE:-cuda:0}"
WANDB_MODE="${WANDB_MODE:-online}"
PROJECT_NAME="${PROJECT_NAME:-HTS-WM-Baselines}"
RUNNING_STEPS="${RUNNING_STEPS:-100000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-2000}"
REPLAY_RATIO="${REPLAY_RATIO:-1}"
BUFFER_SIZE="${BUFFER_SIZE:-1000000}"
START_TRAINING="${START_TRAINING:-1024}"
BATCH_SIZE="${BATCH_SIZE:-16}"
SEQ_LEN="${SEQ_LEN:-64}"
PARALLELS="${PARALLELS:-1}"
BENCHMARK="${BENCHMARK:-1}"
TEST_EPISODE="${TEST_EPISODE:-3}"
CHECKPOINT_RULE="${CHECKPOINT_RULE:-best}"
SEED="${SEED:-1}"
RUN_NAME="${RUN_NAME:-XuanCe-HarmonyDream-${ROM_NAME}-seed${SEED}-${RUNNING_STEPS}steps-rr${REPLAY_RATIO}}"

LOG_DIR="${LOG_DIR:-logs/dreamer_v3/harmonydream/atari100k/${ROM_NAME}/}"
MODEL_DIR="${MODEL_DIR:-models/dreamer_v3/harmonydream/atari100k/${ROM_NAME}/}"
RUN_LOG_DIR="${RUN_LOG_DIR:-${REPO_ROOT}/logs/training_scripts/baselines}"
mkdir -p "${RUN_LOG_DIR}" "${REPO_ROOT}/${LOG_DIR}" "${REPO_ROOT}/${MODEL_DIR}"

(
  cd "${REPO_ROOT}"
  "${PYTHON_BIN}" examples/dreamer_v3/dreamer_v3_atari.py \
    --env-id "${ENV_ID}" \
    --device "${DEVICE}" \
    --seed "${SEED}" \
    --harmony True \
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
    --test-episode "${TEST_EPISODE}" \
    --checkpoint-rule "${CHECKPOINT_RULE}" \
    --log-dir "${LOG_DIR}" \
    --model-dir "${MODEL_DIR}"
) 2>&1 | tee "${RUN_LOG_DIR}/${RUN_NAME}.log"
