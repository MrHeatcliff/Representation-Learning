#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"

ENV_ID="${ENV_ID:-ALE/Breakout-v5}"
ROM_NAME="${ENV_ID##*/}"
DEVICE="${DEVICE:-cuda:0}"
SEED="${SEED:-1}"
WANDB_MODE="${WANDB_MODE:-online}"
PROJECT_NAME="${PROJECT_NAME:-HTS-WM-Baselines}"
RUNNING_STEPS="${RUNNING_STEPS:-100000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-2000}"
REPLAY_RATIO="${REPLAY_RATIO:-0.125}"
BUFFER_SIZE="${BUFFER_SIZE:-1000000}"
START_TRAINING="${START_TRAINING:-1024}"
BATCH_SIZE="${BATCH_SIZE:-16}"
SEQ_LEN="${SEQ_LEN:-64}"
PARALLELS="${PARALLELS:-1}"
BENCHMARK="${BENCHMARK:-1}"
TEST_EPISODE="${TEST_EPISODE:-3}"
CHECKPOINT_RULE="${CHECKPOINT_RULE:-best}"
EVAL_PROTOCOL="${EVAL_PROTOCOL:-final}"
RENDER_EVAL_VIDEO="${RENDER_EVAL_VIDEO:-false}"
RENDER_INTERMEDIATE_VIDEO="${RENDER_INTERMEDIATE_VIDEO:-false}"

BASE_CONFIG="${BASE_CONFIG:-examples/hierarchical_dreamer/config/atari100k_two_phase.yaml}"
RUN_NAME="${RUN_NAME:-tsae-style-atari100k-${ROM_NAME}-seed${SEED}}"
GENERATED_CONFIG="${GENERATED_CONFIG:-${REPO_ROOT}/logs/generated_configs/${RUN_NAME}.yaml}"
LOG_DIR="${LOG_DIR:-logs/hierarchical-dreamer/baselines/tsae_style/${ROM_NAME}/seed${SEED}/}"
MODEL_DIR="${MODEL_DIR:-models/hierarchical-dreamer/baselines/tsae_style/${ROM_NAME}/seed${SEED}/}"
RUN_LOG_DIR="${RUN_LOG_DIR:-${REPO_ROOT}/logs/training_scripts/baselines}"

mkdir -p "$(dirname "${GENERATED_CONFIG}")" "${REPO_ROOT}/${LOG_DIR}" "${REPO_ROOT}/${MODEL_DIR}" "${RUN_LOG_DIR}"

(
  cd "${REPO_ROOT}"
  "${PYTHON_BIN}" examples/hierarchical_dreamer/make_ablation_config.py \
    --base "${BASE_CONFIG}" \
    --output "${GENERATED_CONFIG}" \
    --set training_regime.name=tsae_style \
    --set hierarchical_latent.ablation_name=tsae_style_coarse_temporal \
    --set hierarchical_latent.num_heads=2 \
    --set hierarchical_latent.head_dim=96 \
    --set hierarchical_latent.activation=relu \
    --set hierarchical_latent.sparsity_topk='[20, 20]' \
    --set hierarchical_latent.reconstruction.betas='[0.2, 0.8]' \
    --set hierarchical_latent.sparse_dynamics.strides='[2, 1]' \
    --set hierarchical_latent.sparse_dynamics.alphas='[0.0, 0.0]' \
    --set hierarchical_latent.sparse_dynamics.require_strict_decreasing=true \
    --set hierarchical_latent.temporal_consistency.mode=contrastive \
    --set hierarchical_latent.temporal_consistency.projector_type=none \
    --set hierarchical_latent.temporal_consistency.projection_dim=96 \
    --set hierarchical_latent.temporal_consistency.positive_stride=1 \
    --set hierarchical_latent.variance_covariance.mode=both \
    --set hierarchical_latent.sparsity.mode=level_batch_topk \
    --set hierarchical_latent.sparsity.gammas='[0.0, 0.0]' \
    --set hierarchical_latent.loss_weights.hierarchical=1.0 \
    --set hierarchical_latent.loss_weights.sparse_dynamics=0.0 \
    --set hierarchical_latent.loss_weights.temporal=0.1 \
    --set hierarchical_latent.loss_weights.variance_covariance=0.1 \
    --set hierarchical_latent.loss_weights.sparsity=0.0 \
    --set seed="${SEED}"

  "${PYTHON_BIN}" examples/hierarchical_dreamer/hierarchical_dreamer_atari.py \
    --config-file "${GENERATED_CONFIG}" \
    --env-id "${ENV_ID}" \
    --device "${DEVICE}" \
    --seed "${SEED}" \
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
    --render-eval-video "${RENDER_EVAL_VIDEO}" \
    --render-intermediate-video "${RENDER_INTERMEDIATE_VIDEO}" \
    --checkpoint-rule "${CHECKPOINT_RULE}" \
    --eval-protocol "${EVAL_PROTOCOL}" \
    --log-dir "${LOG_DIR}" \
    --model-dir "${MODEL_DIR}"
) 2>&1 | tee "${RUN_LOG_DIR}/${RUN_NAME}.log"
