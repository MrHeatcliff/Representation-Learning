#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
EAWM_ROOT="${EAWM_ROOT:-${REPO_ROOT}/external_baselines/EAWM}"
EASIMULUS_ROOT="${EASIMULUS_ROOT:-${EAWM_ROOT}/EASimulus}"
PYTHON_BIN="${PYTHON_BIN:-python}"

GAME="${GAME:-Breakout}"
SEED="${SEED:-0}"
DEVICE="${DEVICE:-cuda:0}"
RUN_NAME="${RUN_NAME:-easimulus-eawm-atari100k-${GAME}-seed${SEED}}"
PROJECT_NAME="${PROJECT_NAME:-HTS-WM-Baselines}"
WANDB_MODE="${WANDB_MODE:-online}"
OUTPUTS_DIR="${OUTPUTS_DIR:-${REPO_ROOT}/logs/external_baselines/eawm/easimulus/${RUN_NAME}}"
EPOCHS="${EPOCHS:-600}"
TRAIN_STEPS_PER_EPOCH="${TRAIN_STEPS_PER_EPOCH:-200}"
EVAL_EVERY="${EVAL_EVERY:-50}"
EVENT_PRED="${EVENT_PRED:-True}"
GES="${GES:-True}"
WITH_LPIPS="${WITH_LPIPS:-True}"
RUN_LOG_DIR="${RUN_LOG_DIR:-${REPO_ROOT}/logs/training_scripts/baselines}"

if [[ ! -d "${EASIMULUS_ROOT}" ]]; then
  echo "Missing EASimulus repo at ${EASIMULUS_ROOT}" >&2
  exit 1
fi

mkdir -p "${OUTPUTS_DIR}" "${RUN_LOG_DIR}"

(
  cd "${EASIMULUS_ROOT}"
  "${PYTHON_BIN}" src/main.py \
    tokenizer.image.with_lpips="${WITH_LPIPS}" \
    benchmark=atari \
    env.train.id="${GAME}NoFrameskip-v4" \
    env.test.id="${GAME}NoFrameskip-v4" \
    common.seed="${SEED}" \
    common.device="${DEVICE}" \
    common.epochs="${EPOCHS}" \
    collection.train.config.num_steps="${TRAIN_STEPS_PER_EPOCH}" \
    evaluation.every="${EVAL_EVERY}" \
    world_model.event_pred="${EVENT_PRED}" \
    world_model.ges="${GES}" \
    wandb.mode="${WANDB_MODE}" \
    wandb.project="${PROJECT_NAME}" \
    wandb.name="${RUN_NAME}" \
    wandb.group=eawm-easimulus-atari100k \
    outputs_dir_path="${OUTPUTS_DIR}"
) 2>&1 | tee "${RUN_LOG_DIR}/${RUN_NAME}.log"
