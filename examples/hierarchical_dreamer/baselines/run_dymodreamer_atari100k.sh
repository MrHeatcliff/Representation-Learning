#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

DYMO_ROOT="${DYMO_ROOT:-${REPO_ROOT}/external_baselines/DyMoDreamer}"
PYTHON_BIN="${PYTHON_BIN:-python}"

GAME="${GAME:-breakout}"
SEED="${SEED:-0}"
DEVICE="${DEVICE:-cuda:0}"
RUN_NAME="${RUN_NAME:-dymodreamer-atari100k-${GAME}-seed${SEED}}"
LOG_ROOT="${LOG_ROOT:-${REPO_ROOT}/logs/external_baselines/dymodreamer}"
LOGDIR="${LOGDIR:-${LOG_ROOT}/${RUN_NAME}}"

STEPS="${STEPS:-4e5}"
TRAIN_RATIO="${TRAIN_RATIO:-1024}"
EVAL_EVERY="${EVAL_EVERY:-1e4}"
EVAL_EPISODE_NUM="${EVAL_EPISODE_NUM:-100}"
PREFILL="${PREFILL:-2500}"
PRETRAIN="${PRETRAIN:-100}"
COMPILE="${COMPILE:-False}"
VIDEO_PRED_LOG="${VIDEO_PRED_LOG:-True}"

if [ ! -f "${DYMO_ROOT}/dreamer.py" ]; then
  echo "DyMoDreamer repo not found at ${DYMO_ROOT}."
  echo "Clone it with:"
  echo "  git clone https://github.com/Ultraman-Tiga1/DyMoDreamer.git ${DYMO_ROOT}"
  exit 1
fi

# Upstream imports env modules as envs.atari/envs.wrappers, but the repository
# currently stores those files at the root. Build a lightweight package view at
# runtime without modifying source files.
mkdir -p "${DYMO_ROOT}/envs"
touch "${DYMO_ROOT}/envs/__init__.py"
for module in atari crafter dmc dmlab wrappers; do
  ln -sfn "../${module}.py" "${DYMO_ROOT}/envs/${module}.py"
done

mkdir -p "${LOGDIR}"

cd "${DYMO_ROOT}"
"${PYTHON_BIN}" dreamer.py \
  --configs atari100k \
  --seed "${SEED}" \
  --logdir "${LOGDIR}" \
  --device "${DEVICE}" \
  --task "atari_${GAME}" \
  --steps "${STEPS}" \
  --train_ratio "${TRAIN_RATIO}" \
  --eval_every "${EVAL_EVERY}" \
  --eval_episode_num "${EVAL_EPISODE_NUM}" \
  --prefill "${PREFILL}" \
  --pretrain "${PRETRAIN}" \
  --compile "${COMPILE}" \
  --video_pred_log "${VIDEO_PRED_LOG}"
