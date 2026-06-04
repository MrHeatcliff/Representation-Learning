#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

SGF_ROOT="${SGF_ROOT:-${REPO_ROOT}/external_baselines/sgf}"
PYTHON_BIN="${PYTHON_BIN:-python}"

GAME="${GAME:-Breakout}"
SEED="${SEED:-1}"
DEVICE="${DEVICE:-cuda:0}"
RUN_NAME="${RUN_NAME:-sgf-atari100k-${GAME}-seed${SEED}}"
PROJECT_NAME="${PROJECT_NAME:-HTS-WM-Baselines}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_NOTES="${WANDB_NOTES:-external-code SGF Atari100K baseline}"

BASE_CONFIG="${BASE_CONFIG:-${SGF_ROOT}/configs/default.yaml}"
LOG_ROOT="${LOG_ROOT:-${REPO_ROOT}/logs/external_baselines/sgf}"
RUN_DIR="${RUN_DIR:-${LOG_ROOT}/${RUN_NAME}}"
CONFIG_FILE="${CONFIG_FILE:-${RUN_DIR}/config.yaml}"

ENV_STEPS="${ENV_STEPS:-100000}"
INIT_STEPS="${INIT_STEPS:-5000}"
EVAL_EVERY="${EVAL_EVERY:-2500}"
EVAL_EPISODES="${EVAL_EPISODES:-20}"
FINAL_EVAL_EPISODES="${FINAL_EVAL_EPISODES:-100}"
WM_BATCH_SIZE="${WM_BATCH_SIZE:-1024}"
AGENT_BATCH_SIZE="${AGENT_BATCH_SIZE:-3072}"
AGENT_EVAL="${AGENT_EVAL:-all}"
WM_EVAL="${WM_EVAL:-none}"
AMP="${AMP:-False}"
COMPILE="${COMPILE:-False}"
SAVE="${SAVE:-False}"

if [ ! -f "${SGF_ROOT}/src/main.py" ]; then
  echo "SGF repo not found at ${SGF_ROOT}."
  echo "Clone it with:"
  echo "  git clone https://github.com/jrobine/sgf.git ${SGF_ROOT}"
  exit 1
fi

mkdir -p "${RUN_DIR}"

"${PYTHON_BIN}" - "${BASE_CONFIG}" "${CONFIG_FILE}" \
  "${ENV_STEPS}" "${INIT_STEPS}" "${EVAL_EVERY}" "${EVAL_EPISODES}" "${FINAL_EVAL_EPISODES}" \
  "${WM_BATCH_SIZE}" "${AGENT_BATCH_SIZE}" <<'PY'
from pathlib import Path
import sys
from ruamel.yaml import YAML

base, output = Path(sys.argv[1]), Path(sys.argv[2])
env_steps = int(float(sys.argv[3]))
init_steps = int(float(sys.argv[4]))
eval_every = int(float(sys.argv[5]))
eval_episodes = int(float(sys.argv[6]))
final_eval_episodes = int(float(sys.argv[7]))
wm_batch_size = int(float(sys.argv[8]))
agent_batch_size = int(float(sys.argv[9]))

yaml = YAML()
with base.open("r") as file:
    config = yaml.load(file)

config["trainer"]["env_steps"] = env_steps
config["trainer"]["init_steps"] = init_steps
config["trainer"]["eval_every"] = eval_every
config["trainer"]["agent_trainer"]["eval_episodes"] = eval_episodes
config["trainer"]["agent_trainer"]["final_eval_episodes"] = final_eval_episodes
config["trainer"]["wm_trainer"]["batch_size"] = wm_batch_size
config["trainer"]["agent_trainer"]["batch_size"] = agent_batch_size

output.parent.mkdir(parents=True, exist_ok=True)
with output.open("w") as file:
    yaml.dump(config, file)
PY

cd "${SGF_ROOT}"

args=(
  src/main.py
  --device "${DEVICE}"
  --game "${GAME}"
  --project "${PROJECT_NAME}"
  --mode "${WANDB_MODE}"
  --notes "${WANDB_NOTES}; run=${RUN_NAME}"
  --config "${CONFIG_FILE}"
  --seed "${SEED}"
  --wm_eval "${WM_EVAL}"
  --agent_eval "${AGENT_EVAL}"
)

if [ "${AMP}" = "True" ]; then
  args+=(--amp)
fi
if [ "${COMPILE}" = "True" ]; then
  args+=(--compile)
fi
if [ "${SAVE}" = "True" ]; then
  args+=(--save)
fi

"${PYTHON_BIN}" "${args[@]}"
