# DyMoDreamer Baseline

This baseline uses the upstream DyMoDreamer repository as an external baseline
instead of a guessed reimplementation inside XuanCe.

## Upstream

- Repository: `https://github.com/Ultraman-Tiga1/DyMoDreamer.git`
- Local clone: `external_baselines/DyMoDreamer`
- Inspected commit: `c40e4f428c5f7e6cd68acb2f576bf3a1072db4d9`
- License: no license file found in the cloned repository.

## Current Assessment

Runnable path: `PARTIAL`.

The repository contains a complete PyTorch training entrypoint and an Atari100K
config, but the upstream repo is not packaged and its README does not list
dependencies. It also imports Atari/DMC wrappers as `envs.*` while storing the
files at repository root. The wrapper script creates a lightweight `envs/`
package view at runtime using symlinks.

## Atari100K Config Found Upstream

`configs.yaml` contains:

- `atari100k.steps: 4e5`
- `envs: 1`
- `action_repeat: 4`
- `train_ratio: 1024`
- `eval_episode_num: 100`
- `actor.dist: onehot`
- `imag_gradient: reinforce`
- `sticky: False` via config key typo `stickey: False`
- `lives: unused`
- `noops: 30`
- `resize: opencv`
- `actions: needed`
- `time_limit: 108000`

The training code divides `steps`, `eval_every`, `log_every`, and `time_limit`
by `action_repeat`, so `steps=4e5` corresponds to roughly 100K agent steps.

## Setup

Use a separate environment. Do not install these dependencies into the XuanCe
`.venv` unless you intentionally want to share the environment.

```bash
conda create -n dymodreamer python=3.10 -y
conda activate dymodreamer

python -m pip install --upgrade pip setuptools wheel

# Install PyTorch matching your CUDA/driver. Example for CUDA 12.6:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# Core dependencies inferred from imports.
# DyMoDreamer uses ruamel.yaml.safe_load(), which was removed in ruamel.yaml 0.18+.
pip install numpy "ruamel.yaml<0.18" termcolor opencv-python pillow tensorboard gym==0.19.0

# If gym==0.19.0 fails with modern packaging:
python -m pip install "pip<24" "setuptools==57.5.0" "wheel==0.38.4"
pip install --no-build-isolation gym==0.19.0

# Atari support.
pip install "gym[atari]==0.19.0" --no-build-isolation
pip install autorom "ale-py<0.8"
AutoROM --accept-license
```

Check:

```bash
python - <<'PY'
import torch, gym, ruamel.yaml, termcolor, cv2
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
env = gym.make("PongNoFrameskip-v4")
print(env.observation_space, env.action_space)
env.close()
PY
```

## Smoke Run

```bash
cd /mnt/disk1/backup_user/dat.tt2/xuance
conda activate dymodreamer

GAME=breakout \
SEED=0 \
DEVICE=cuda:0 \
RUN_NAME=dymodreamer-smoke-breakout-seed0 \
STEPS=1000 \
TRAIN_RATIO=16 \
EVAL_EVERY=500 \
EVAL_EPISODE_NUM=1 \
PREFILL=10 \
PRETRAIN=1 \
COMPILE=False \
VIDEO_PRED_LOG=False \
examples/hierarchical_dreamer/baselines/run_dymodreamer_atari100k.sh
```

## Full Atari100K Run

```bash
cd /mnt/disk1/backup_user/dat.tt2/xuance
conda activate dymodreamer

GAME=breakout \
SEED=0 \
DEVICE=cuda:0 \
RUN_NAME=dymodreamer-atari100k-breakout-seed0 \
STEPS=4e5 \
TRAIN_RATIO=1024 \
EVAL_EVERY=1e4 \
EVAL_EPISODE_NUM=100 \
COMPILE=False \
VIDEO_PRED_LOG=True \
examples/hierarchical_dreamer/baselines/run_dymodreamer_atari100k.sh
```

Logs are written to:

```text
logs/external_baselines/dymodreamer/<RUN_NAME>/
```

Expected artifacts:

- `metrics.jsonl`
- TensorBoard events
- `latest.pt`
- train/eval episode `.npz` files

## Caveats

- This is an external-code baseline, not a same-code XuanCe ablation.
- Upstream has no license file in the cloned repo.
- Upstream README is minimal, so dependency versions are inferred from imports.
- The wrapper keeps `COMPILE=False` by default because `torch.compile` can add
  significant startup overhead and may fail in smoke tests.
- The upstream code does not include W&B logging; use `metrics.jsonl` and
  TensorBoard logs for aggregation.
