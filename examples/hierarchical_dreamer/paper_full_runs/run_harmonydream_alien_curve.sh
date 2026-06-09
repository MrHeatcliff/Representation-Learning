#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"

cd "${REPO_ROOT}"

METHODS="${METHODS:-dreamer,harmony}"
SEEDS="${SEEDS:-0}"
ENV_ID="${ENV_ID:-ALE/Alien-v5}"
ROM_NAME="${ENV_ID##*/}"
DEVICE="${DEVICE:-cuda:0}"
WANDB_MODE="${WANDB_MODE:-online}"
PROJECT_NAME="${PROJECT_NAME:-HTS-WM-HarmonyDream-Alien-Curve}"

OBS_TYPE="${OBS_TYPE:-rgb}"
NUM_STACK="${NUM_STACK:-1}"
FRAME_SKIP="${FRAME_SKIP:-4}"
REPEAT_ACTION_PROBABILITY="${REPEAT_ACTION_PROBABILITY:-0.0}"
CLIP_REWARD="${CLIP_REWARD:-false}"
EPISODIC_LIFE="${EPISODIC_LIFE:-false}"
IMG_SIZE_0="${IMG_SIZE_0:-64}"
IMG_SIZE_1="${IMG_SIZE_1:-64}"
if [[ -n "${MODEL_SIZE+x}" && -z "${HARMONY_MODEL_SIZE+x}" && "${MODEL_SIZE}" != "size12m" ]]; then
  echo "Ignoring inherited MODEL_SIZE=${MODEL_SIZE}; HarmonyDream Alien curve uses HARMONY_MODEL_SIZE=${HARMONY_MODEL_SIZE:-size12m}."
fi
MODEL_SIZE="${HARMONY_MODEL_SIZE:-size12m}"

RUNNING_STEPS="${RUNNING_STEPS:-100000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-5000}"
REPLAY_RATIO="${REPLAY_RATIO:-0.25}"
BUFFER_SIZE="${BUFFER_SIZE:-1000000}"
START_TRAINING="${START_TRAINING:-1024}"
BATCH_SIZE="${BATCH_SIZE:-16}"
SEQ_LEN="${SEQ_LEN:-64}"
PARALLELS="${PARALLELS:-1}"
BENCHMARK="${BENCHMARK:-1}"
TEST_EPISODE="${TEST_EPISODE:-100}"
INTERMEDIATE_TEST_EPISODE="${INTERMEDIATE_TEST_EPISODE:-100}"
CHECKPOINT_RULE="${CHECKPOINT_RULE:-final}"
EVAL_PROTOCOL="${EVAL_PROTOCOL:-periodic}"
RENDER_EVAL_VIDEO="${RENDER_EVAL_VIDEO:-true}"
RENDER_INTERMEDIATE_VIDEO="${RENDER_INTERMEDIATE_VIDEO:-false}"

RUN_GROUP="${RUN_GROUP:-harmonydream_alien_curve_$(date +%Y%m%d_%H%M%S)}"
RUN_LOG_ROOT="${RUN_LOG_ROOT:-${REPO_ROOT}/logs/training_scripts/${RUN_GROUP}}"
PREPROCESS_TAG="${OBS_TYPE}-${IMG_SIZE_0}x${IMG_SIZE_1}-stack${NUM_STACK}-repeat${FRAME_SKIP}-${MODEL_SIZE}"
mkdir -p "${RUN_LOG_ROOT}"

contains_method() {
  local needle="$1"
  [[ ",${METHODS}," == *",${needle},"* || "${METHODS}" == "all" ]]
}

run_dreamer() {
  local seed="$1"
  local run_name="DreamerV3-reproduced-${ROM_NAME}-${PREPROCESS_TAG}-seed${seed}-${RUNNING_STEPS}steps"
  local log_file="${RUN_LOG_ROOT}/${run_name}.log"
  (
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
      --seed "${seed}" \
      --harmony false \
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
      --condition "dreamerv3_reproduced" \
      --log-dir "logs/harmonydream-alien-curve/dreamerv3/${PREPROCESS_TAG}/${ROM_NAME}/seed${seed}/" \
      --model-dir "models/harmonydream-alien-curve/dreamerv3/${PREPROCESS_TAG}/${ROM_NAME}/seed${seed}/"
  ) 2>&1 | tee "${log_file}"
}

run_harmony() {
  local seed="$1"
  local run_name="HarmonyDream-${ROM_NAME}-${PREPROCESS_TAG}-seed${seed}-${RUNNING_STEPS}steps"
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
  SEED="${seed}" \
  DEVICE="${DEVICE}" \
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
  TEST_EPISODE="${TEST_EPISODE}" \
  INTERMEDIATE_TEST_EPISODE="${INTERMEDIATE_TEST_EPISODE}" \
  RENDER_EVAL_VIDEO="${RENDER_EVAL_VIDEO}" \
  RENDER_INTERMEDIATE_VIDEO="${RENDER_INTERMEDIATE_VIDEO}" \
  CHECKPOINT_RULE="${CHECKPOINT_RULE}" \
  EVAL_PROTOCOL="${EVAL_PROTOCOL}" \
  PYTHON_BIN="${PYTHON_BIN}" \
  LOG_DIR="logs/harmonydream-alien-curve/harmony/${PREPROCESS_TAG}/${ROM_NAME}/seed${seed}/" \
  MODEL_DIR="models/harmonydream-alien-curve/harmony/${PREPROCESS_TAG}/${ROM_NAME}/seed${seed}/" \
  RUN_LOG_DIR="${RUN_LOG_ROOT}" \
  examples/hierarchical_dreamer/baselines/run_xuance_harmonydream_atari100k.sh
}

echo "Run group: ${RUN_GROUP}"
echo "Methods: ${METHODS}"
echo "Seeds: ${SEEDS}"
echo "Environment: ${ENV_ID}"
echo "Protocol: ${PREPROCESS_TAG}, replay_ratio=${REPLAY_RATIO}, eval_every=${EVAL_INTERVAL}, eval_eps=${TEST_EPISODE}"
echo "Logs: ${RUN_LOG_ROOT}"

IFS=',' read -ra SEED_LIST <<< "${SEEDS}"
for seed in "${SEED_LIST[@]}"; do
  if contains_method dreamer; then
    run_dreamer "${seed}"
  fi
  if contains_method harmony; then
    run_harmony "${seed}"
  fi
done
