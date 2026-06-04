#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
EAWM_ROOT="${EAWM_ROOT:-${REPO_ROOT}/external_baselines/EAWM}"
EADREAM_ROOT="${EADREAM_ROOT:-${EAWM_ROOT}/EADream}"
PYTHON_BIN="${PYTHON_BIN:-python}"

GAME="${GAME:-breakout}"
SEED="${SEED:-0}"
DEVICE="${DEVICE:-cuda:0}"
RUN_NAME="${RUN_NAME:-eadream-atari100k-${GAME}-seed${SEED}}"
STEPS="${STEPS:-4e5}"
TRAIN_RATIO="${TRAIN_RATIO:-1024}"
EVAL_EVERY="${EVAL_EVERY:-3e4}"
LOG_EVERY="${LOG_EVERY:-3e4}"
EVAL_EPISODE_NUM="${EVAL_EPISODE_NUM:-100}"
COMPILE="${COMPILE:-False}"
VIDEO_PRED_LOG="${VIDEO_PRED_LOG:-True}"
LOG_ROOT="${LOG_ROOT:-${REPO_ROOT}/logs/external_baselines/eawm/eadream}"
RUN_LOG_DIR="${RUN_LOG_DIR:-${REPO_ROOT}/logs/training_scripts/baselines}"

if [[ ! -d "${EADREAM_ROOT}" ]]; then
  echo "Missing EADream repo at ${EADREAM_ROOT}" >&2
  exit 1
fi

mkdir -p "${LOG_ROOT}" "${RUN_LOG_DIR}"
LOG_DIR="${LOG_DIR:-${LOG_ROOT}/${RUN_NAME}}"

(
  cd "${EADREAM_ROOT}"
  "${PYTHON_BIN}" dreamer.py \
    --configdir configsc.yaml \
    --configs atari100k \
    --task "atari_${GAME}" \
    --seed "${SEED}" \
    --logdir "${LOG_DIR}" \
    --device "${DEVICE}" \
    --steps "${STEPS}" \
    --train_ratio "${TRAIN_RATIO}" \
    --eval_every "${EVAL_EVERY}" \
    --log_every "${LOG_EVERY}" \
    --eval_episode_num "${EVAL_EPISODE_NUM}" \
    --compile "${COMPILE}" \
    --video_pred_log "${VIDEO_PRED_LOG}"
) 2>&1 | tee "${RUN_LOG_DIR}/${RUN_NAME}.log"
