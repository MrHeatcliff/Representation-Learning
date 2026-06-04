#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
TSAE_ROOT="${TSAE_ROOT:-${REPO_ROOT}/external_baselines/temporal-saes}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ ! -d "${TSAE_ROOT}" ]]; then
  echo "Missing T-SAE repo at ${TSAE_ROOT}" >&2
  exit 1
fi

export PYTHONPATH="${TSAE_ROOT}:${TSAE_ROOT}/dictionary_learning:${PYTHONPATH:-}"

"${PYTHON_BIN}" - <<'PY'
import torch
import transformers
import datasets
import nnsight
from dictionary_learning.trainers import (
    TemporalMatryoshkaBatchTopKSAE,
    TemporalMatryoshkaBatchTopKTrainer,
)

print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("transformers", transformers.__version__)
print("datasets", datasets.__version__)
print("nnsight", nnsight.__version__)
print("T-SAE trainer import OK")
print(TemporalMatryoshkaBatchTopKSAE.__name__, TemporalMatryoshkaBatchTopKTrainer.__name__)
PY
