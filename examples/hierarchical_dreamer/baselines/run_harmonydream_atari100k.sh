#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

HARMONY_ROOT="${HARMONY_ROOT:-${REPO_ROOT}/external_baselines/HarmonyDream}"
HARMONY_JAX_ROOT="${HARMONY_JAX_ROOT:-${HARMONY_ROOT}/dreamerv3-jax}"
PYTHON_BIN="${PYTHON_BIN:-python}"

GAME="${GAME:-breakout}"
SEED="${SEED:-0}"
DEVICE_ID="${DEVICE_ID:-0}"
RUN_NAME="${RUN_NAME:-harmonydream-atari100k-${GAME}-seed${SEED}}"
LOG_ROOT="${LOG_ROOT:-${REPO_ROOT}/logs/external_baselines/harmonydream}"
LOGDIR="${LOGDIR:-${LOG_ROOT}/${RUN_NAME}}"
HARMONY="${HARMONY:-True}"

STEPS="${STEPS:-1.01e5}"
TRAIN_RATIO="${TRAIN_RATIO:-1024}"
EVAL_EVERY="${EVAL_EVERY:-5e3}"
EVAL_EPS="${EVAL_EPS:-100}"
JAX_PREALLOC="${JAX_PREALLOC:-False}"
JAX_PRECISION="${JAX_PRECISION:-float32}"

if [ ! -d "${HARMONY_JAX_ROOT}/dreamerv3" ]; then
  echo "HarmonyDream repo not found at ${HARMONY_ROOT}."
  echo "Clone it with:"
  echo "  git clone https://github.com/thuml/HarmonyDream.git ${HARMONY_ROOT}"
  exit 1
fi

mkdir -p "${LOGDIR}"

cd "${HARMONY_JAX_ROOT}"
CUDA_VISIBLE_DEVICES="${DEVICE_ID}" "${PYTHON_BIN}" dreamerv3/train.py \
  --logdir "${LOGDIR}" \
  --configs atari100k \
  --seed "${SEED}" \
  --task "atari_${GAME}" \
  --harmony "${HARMONY}" \
  --run.steps "${STEPS}" \
  --run.train_ratio "${TRAIN_RATIO}" \
  --run.eval_every "${EVAL_EVERY}" \
  --run.eval_eps "${EVAL_EPS}" \
  --jax.prealloc "${JAX_PREALLOC}" \
  --jax.precision "${JAX_PRECISION}"
