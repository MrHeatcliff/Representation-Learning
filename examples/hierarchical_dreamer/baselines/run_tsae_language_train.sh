#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
TSAE_ROOT="${TSAE_ROOT:-${REPO_ROOT}/external_baselines/temporal-saes}"
PYTHON_BIN="${PYTHON_BIN:-python}"

MODEL="${MODEL:-EleutherAI/pythia-160m}"
LAYER="${LAYER:-8}"
RUN_NAME="${RUN_NAME:-tsae-${MODEL//\//_}-layer${LAYER}}"
MAX_STEPS="${MAX_STEPS:-200000}"
K="${K:-20}"
TEMP_ALPHA="${TEMP_ALPHA:-0.1}"
TEMPORAL="${TEMPORAL:-r}"
WANDB_PROJECT="${WANDB_PROJECT:-temporal-saes-baseline}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
DISABLE_WANDB="${DISABLE_WANDB:-false}"
SAVE_ROOT="${SAVE_ROOT:-${REPO_ROOT}/logs/external_baselines/tsae_language}"

if [[ ! -d "${TSAE_ROOT}" ]]; then
  echo "Missing T-SAE repo at ${TSAE_ROOT}" >&2
  exit 1
fi

mkdir -p "${SAVE_ROOT}"
export PYTHONPATH="${TSAE_ROOT}:${TSAE_ROOT}/dictionary_learning:${PYTHONPATH:-}"

args=(
  dictionary_learning/dictionary_learning/train_temporal.py
  --model "${MODEL}"
  --layer "${LAYER}"
  --run-name "${RUN_NAME}"
  --max-steps "${MAX_STEPS}"
  --k "${K}"
  --temp_alpha "${TEMP_ALPHA}"
  --temporal "${TEMPORAL}"
  --save-path "${SAVE_ROOT}"
  --wandb-project "${WANDB_PROJECT}"
)

if [[ -n "${WANDB_ENTITY}" ]]; then
  args+=(--wandb-entity "${WANDB_ENTITY}")
fi

if [[ "${DISABLE_WANDB}" == "true" || "${DISABLE_WANDB}" == "1" ]]; then
  args+=(--disable-wandb)
fi

cd "${TSAE_ROOT}"
"${PYTHON_BIN}" "${args[@]}"
