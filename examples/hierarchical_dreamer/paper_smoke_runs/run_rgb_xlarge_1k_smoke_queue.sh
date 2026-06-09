#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"

cd "${REPO_ROOT}"

export ENV_ID="${ENV_ID:-ALE/Breakout-v5}"
ROM_NAME="${ENV_ID##*/}"
export DEVICE="${DEVICE:-cuda:0}"
export SEED="${SEED:-1}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export PROJECT_NAME="${PROJECT_NAME:-HTS-WM-RGB-Small-Smoke}"
export RUNNING_STEPS="${RUNNING_STEPS:-1000}"
export EVAL_INTERVAL="${EVAL_INTERVAL:-500}"
export REPLAY_RATIO="${REPLAY_RATIO:-1.0}"
export BUFFER_SIZE="${BUFFER_SIZE:-1000000}"
export START_TRAINING="${START_TRAINING:-100}"
export BATCH_SIZE="${BATCH_SIZE:-16}"
export SEQ_LEN="${SEQ_LEN:-64}"
export PARALLELS="${PARALLELS:-1}"
export BENCHMARK="${BENCHMARK:-1}"
export CHECKPOINT_RULE="${CHECKPOINT_RULE:-final}"
export EVAL_PROTOCOL="${EVAL_PROTOCOL:-final}"
export TEST_EPISODE="${TEST_EPISODE:-1}"
export INTERMEDIATE_TEST_EPISODE="${INTERMEDIATE_TEST_EPISODE:-1}"
export RENDER_EVAL_VIDEO="${RENDER_EVAL_VIDEO:-false}"
export RENDER_INTERMEDIATE_VIDEO="${RENDER_INTERMEDIATE_VIDEO:-false}"
export OBS_TYPE="${OBS_TYPE:-rgb}"
export NUM_STACK="${NUM_STACK:-1}"
export FRAME_SKIP="${FRAME_SKIP:-4}"
export REPEAT_ACTION_PROBABILITY="${REPEAT_ACTION_PROBABILITY:-0.0}"
export CLIP_REWARD="${CLIP_REWARD:-false}"
export EPISODIC_LIFE="${EPISODIC_LIFE:-false}"
export IMG_SIZE_0="${IMG_SIZE_0:-64}"
export IMG_SIZE_1="${IMG_SIZE_1:-64}"
export MODEL_SIZE="${MODEL_SIZE:-small}"

SMOKE_TAG="${SMOKE_TAG:-rgb-small-1k}"
SMOKE_METHODS="${SMOKE_METHODS:-all}"
RUN_GROUP="${RUN_GROUP:-rgb_small_1k_smoke_$(date +%Y%m%d_%H%M%S)}"
RUN_LOG_ROOT="${RUN_LOG_ROOT:-${REPO_ROOT}/logs/training_scripts/${RUN_GROUP}}"
RUN_ROOT="${RUN_ROOT:-logs/rgb_small_smoke}"
MODEL_ROOT="${MODEL_ROOT:-models/rgb_small_smoke}"
PREPROCESS_TAG="${OBS_TYPE}-${IMG_SIZE_0}x${IMG_SIZE_1}-stack${NUM_STACK}-repeat${FRAME_SKIP}-${MODEL_SIZE}"

mkdir -p "${RUN_LOG_ROOT}"

contains_method() {
  local needle="$1"
  [[ "${SMOKE_METHODS}" == "all" || ",${SMOKE_METHODS}," == *",${needle},"* ]]
}

run_with_log() {
  local run_name="$1"
  shift
  local log_file="${RUN_LOG_ROOT}/${run_name}.log"
  echo "===== RUN ${run_name} ====="
  "$@" 2>&1 | tee "${log_file}"
}

run_dreamer() {
  local run_name="smoke-dreamerv3-${ROM_NAME}-seed${SEED}-${RUNNING_STEPS}steps-${SMOKE_TAG}"
  run_with_log "${run_name}" \
    "${PYTHON_BIN}" examples/dreamer_v3/dreamer_v3_atari.py \
      --env-id "${ENV_ID}" \
      --obs-type "${OBS_TYPE}" \
      --num-stack "${NUM_STACK}" \
      --frame-skip "${FRAME_SKIP}" \
      --repeat-action-probability "${REPEAT_ACTION_PROBABILITY}" \
      --clip-reward "${CLIP_REWARD}" \
      --episodic-life "${EPISODIC_LIFE}" \
      --img-size "${IMG_SIZE_0}" "${IMG_SIZE_1}" \
      --model-size "${MODEL_SIZE}" \
      --device "${DEVICE}" \
      --seed "${SEED}" \
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
      --log-dir "${RUN_ROOT}/${run_name}/" \
      --model-dir "${MODEL_ROOT}/${run_name}/"
}

run_hts_config() {
  local condition="$1"
  local config_file="$2"
  local run_name="smoke-${condition}-${ROM_NAME}-seed${SEED}-${RUNNING_STEPS}steps-${SMOKE_TAG}"
  run_with_log "${run_name}" env \
    CONFIG_FILE="${config_file}" \
    RUN_NAME="${run_name}" \
    ENV_ID="${ENV_ID}" \
    OBS_TYPE="${OBS_TYPE}" \
    NUM_STACK="${NUM_STACK}" \
    FRAME_SKIP="${FRAME_SKIP}" \
    REPEAT_ACTION_PROBABILITY="${REPEAT_ACTION_PROBABILITY}" \
    CLIP_REWARD="${CLIP_REWARD}" \
    EPISODIC_LIFE="${EPISODIC_LIFE}" \
    IMG_SIZE_0="${IMG_SIZE_0}" \
    IMG_SIZE_1="${IMG_SIZE_1}" \
    MODEL_SIZE="${MODEL_SIZE}" \
    DEVICE="${DEVICE}" \
    SEED="${SEED}" \
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
    CHECKPOINT_RULE="${CHECKPOINT_RULE}" \
    EVAL_PROTOCOL="${EVAL_PROTOCOL}" \
    TEST_EPISODE="${TEST_EPISODE}" \
    INTERMEDIATE_TEST_EPISODE="${INTERMEDIATE_TEST_EPISODE}" \
    RENDER_EVAL_VIDEO="${RENDER_EVAL_VIDEO}" \
    RENDER_INTERMEDIATE_VIDEO="${RENDER_INTERMEDIATE_VIDEO}" \
    PYTHON_BIN="${PYTHON_BIN}" \
    LOG_DIR="${RUN_ROOT}/${run_name}/" \
    MODEL_DIR="${MODEL_ROOT}/${run_name}/" \
    RUN_LOG_DIR="${RUN_LOG_ROOT}" \
    examples/hierarchical_dreamer/train_ablation.sh
}

run_xuance_harmony() {
  local run_name="smoke-xuance-harmonydream-${ROM_NAME}-seed${SEED}-${RUNNING_STEPS}steps-${SMOKE_TAG}"
  run_with_log "${run_name}" env \
    RUN_NAME="${run_name}" \
    ENV_ID="${ENV_ID}" \
    OBS_TYPE="${OBS_TYPE}" \
    NUM_STACK="${NUM_STACK}" \
    FRAME_SKIP="${FRAME_SKIP}" \
    REPEAT_ACTION_PROBABILITY="${REPEAT_ACTION_PROBABILITY}" \
    CLIP_REWARD="${CLIP_REWARD}" \
    EPISODIC_LIFE="${EPISODIC_LIFE}" \
    IMG_SIZE_0="${IMG_SIZE_0}" \
    IMG_SIZE_1="${IMG_SIZE_1}" \
    MODEL_SIZE="${MODEL_SIZE}" \
    DEVICE="${DEVICE}" \
    SEED="${SEED}" \
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
    CHECKPOINT_RULE="${CHECKPOINT_RULE}" \
    EVAL_PROTOCOL="${EVAL_PROTOCOL}" \
    TEST_EPISODE="${TEST_EPISODE}" \
    INTERMEDIATE_TEST_EPISODE="${INTERMEDIATE_TEST_EPISODE}" \
    RENDER_EVAL_VIDEO="${RENDER_EVAL_VIDEO}" \
    RENDER_INTERMEDIATE_VIDEO="${RENDER_INTERMEDIATE_VIDEO}" \
    PYTHON_BIN="${PYTHON_BIN}" \
    LOG_DIR="${RUN_ROOT}/${run_name}/" \
    MODEL_DIR="${MODEL_ROOT}/${run_name}/" \
    RUN_LOG_DIR="${RUN_LOG_ROOT}" \
    examples/hierarchical_dreamer/baselines/run_xuance_harmonydream_atari100k.sh
}

run_tsae_style() {
  local run_name="smoke-tsae-style-${ROM_NAME}-seed${SEED}-${RUNNING_STEPS}steps-${SMOKE_TAG}"
  run_with_log "${run_name}" env \
    RUN_NAME="${run_name}" \
    ENV_ID="${ENV_ID}" \
    OBS_TYPE="${OBS_TYPE}" \
    NUM_STACK="${NUM_STACK}" \
    FRAME_SKIP="${FRAME_SKIP}" \
    REPEAT_ACTION_PROBABILITY="${REPEAT_ACTION_PROBABILITY}" \
    CLIP_REWARD="${CLIP_REWARD}" \
    EPISODIC_LIFE="${EPISODIC_LIFE}" \
    IMG_SIZE_0="${IMG_SIZE_0}" \
    IMG_SIZE_1="${IMG_SIZE_1}" \
    MODEL_SIZE="${MODEL_SIZE}" \
    DEVICE="${DEVICE}" \
    SEED="${SEED}" \
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
    CHECKPOINT_RULE="${CHECKPOINT_RULE}" \
    EVAL_PROTOCOL="${EVAL_PROTOCOL}" \
    TEST_EPISODE="${TEST_EPISODE}" \
    INTERMEDIATE_TEST_EPISODE="${INTERMEDIATE_TEST_EPISODE}" \
    RENDER_EVAL_VIDEO="${RENDER_EVAL_VIDEO}" \
    RENDER_INTERMEDIATE_VIDEO="${RENDER_INTERMEDIATE_VIDEO}" \
    PYTHON_BIN="${PYTHON_BIN}" \
    LOG_DIR="${RUN_ROOT}/${run_name}/" \
    MODEL_DIR="${MODEL_ROOT}/${run_name}/" \
    RUN_LOG_DIR="${RUN_LOG_ROOT}" \
    examples/hierarchical_dreamer/baselines/run_tsae_style_atari100k.sh
}

echo "Run group: ${RUN_GROUP}"
echo "Environment: ${ENV_ID}"
echo "Preprocessing/model: ${PREPROCESS_TAG}"
echo "Steps: ${RUNNING_STEPS}, start_training: ${START_TRAINING}, replay_ratio: ${REPLAY_RATIO}"
echo "Methods: ${SMOKE_METHODS}"
echo "W&B mode: ${WANDB_MODE}"
echo "Script logs: ${RUN_LOG_ROOT}"

if contains_method dreamer; then
  run_dreamer
fi
if contains_method hts-full; then
  run_hts_config "hts-full" "examples/hierarchical_dreamer/config/atari100k_two_phase.yaml"
fi
if contains_method flat-sae; then
  run_hts_config "flat-sae" "examples/hierarchical_dreamer/config/generated_configs/flat_sae.yaml"
fi
if contains_method flat-mh; then
  run_hts_config "flat-mh" "examples/hierarchical_dreamer/config/generated_configs/flat_mh.yaml"
fi
if contains_method sgf-style-flat-same-code; then
  run_hts_config "sgf-style-flat-same-code" "examples/hierarchical_dreamer/config/generated_configs/sgf_style_flat_same_code.yaml"
fi
if contains_method recon-only-hierarchy; then
  run_hts_config "recon-only-hierarchy" "examples/hierarchical_dreamer/config/generated_configs/recon_only_hierarchy.yaml"
fi
if contains_method dense-multistride-no-sparse; then
  run_hts_config "dense-multistride-no-sparse" "examples/hierarchical_dreamer/config/generated_configs/dense_multistride_no_sparse.yaml"
fi
if contains_method hts-no-hier; then
  run_hts_config "hts-no-hier" "examples/hierarchical_dreamer/config/generated_configs/hts_no_hier.yaml"
fi
if contains_method hts-no-sdyn; then
  run_hts_config "hts-no-sdyn" "examples/hierarchical_dreamer/config/generated_configs/hts_no_sdyn.yaml"
fi
if contains_method hts-no-temp; then
  run_hts_config "hts-no-temp" "examples/hierarchical_dreamer/config/generated_configs/hts_no_temp.yaml"
fi
if contains_method hts-no-vc; then
  run_hts_config "hts-no-vc" "examples/hierarchical_dreamer/config/generated_configs/hts_no_vc.yaml"
fi

if contains_method larger-flat-param && [[ "${ENV_ID}" == "ALE/Breakout-v5" ]]; then
  run_hts_config "larger-flat-param" "examples/hierarchical_dreamer/config/generated_configs/larger_flat_param_breakout.yaml"
elif contains_method larger-flat-param; then
  echo "Skip larger-flat-param: generated config is Breakout-specific."
fi

if contains_method xuance-harmony; then
  run_xuance_harmony
fi
if contains_method tsae-style; then
  run_tsae_style
fi

echo "RGB small 1K smoke queue finished."
