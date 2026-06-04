# EAWM Baseline Notes

This baseline uses the official EAWM repository as an external-code baseline.

## Source

- Repository: `https://github.com/MarquisDarwin/EAWM.git`
- Local clone: `external_baselines/EAWM`
- Inspected commit: `d37f1ffa6c9dd42c163ed9e29e717bbafa399f37`
- License: GPL-3.0
- Paper: `From Observations to Events: Event-Aware World Models for Reinforcement Learning`

The local clone contains two implementations:

- `EADream`: DreamerV3-style implementation for Atari100K, DMC Vision, and
  DMC-GB2. This is the preferred first baseline for Dreamer-style comparisons.
- `EASimulus`: Simulus-based implementation for Atari100K and Craftax. This has
  native W&B logging and can be used as a secondary official-code baseline.

## Status

`PARTIAL`: source is available locally and launch wrappers are provided. The
dedicated environments still need to be created and smoke-tested before marking
the baseline paper-ready.

## Why A Separate Environment

Do not install EAWM dependencies into the XuanCe `uv` env or the existing
`harmonydream` env. EAWM has two dependency stacks:

- `EADream`: Python 3.9, `gym==0.19.0`, `atari-py==0.2.9`, old
  `ruamel.yaml`, `dm_control==1.0.9`.
- `EASimulus`: Python 3.10, `gymnasium[atari]==1.0.0a2`, `ale-py==0.9.1`,
  Hydra, RetNet package, W&B.

## EADream Atari-Only Environment

```bash
conda create -n eadream python=3.9 -y
conda activate eadream
python -m pip install "pip<24.1" setuptools==60.0.0 wheel

# Use an older torch stack closer to the upstream Dockerfile. This avoids the
# numpy/scipy conflicts caused by torch 2.5+ on Python 3.9.
conda install -y pytorch=2.0.1 torchvision=0.15.2 pytorch-cuda=11.7 -c pytorch -c nvidia -c defaults

cd /mnt/disk1/backup_user/dat.tt2/xuance/external_baselines/EAWM/EADream

# Atari-only dependencies. Do not install the raw requirements.txt for Atari
# runs because it also includes DMC/MemoryMaze dependencies that can trigger a
# MuJoCo source build.
conda install -y -c conda-forge \
  gym=0.19.0 \
  numpy=1.23.5 \
  scipy=1.8 \
  matplotlib=3.5 \
  moviepy=1.0.3 \
  tensorboard \
  ruamel.yaml=0.17.4 \
  einops=0.3.0 \
  xmltodict \
  opencv

python -m pip install \
  atari-py==0.2.9 \
  mujoco==2.3.5 \
  dm-control==1.0.9 \
  dm-env==1.6 \
  crafter==1.8.0 \
  rliable \
  tbparse \
  lpips \
  --timeout 120 \
  --retries 20

python -m atari_py.import_roms "scripts/AtariROMS"
```

If importing from the XuanCe repo root, use the full relative path instead:

```bash
cd /mnt/disk1/backup_user/dat.tt2/xuance
/home/dat.tt2/miniconda3/envs/eadream/bin/python -m atari_py.import_roms \
  external_baselines/EAWM/EADream/scripts/AtariROMS
```

Smoke import:

```bash
cd /mnt/disk1/backup_user/dat.tt2/xuance/external_baselines/EAWM/EADream
python - <<'PY'
import torch, gym, atari_py, cv2, ruamel.yaml
import dreamer
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("gym", gym.__version__)
print("EADream import OK")
PY
```

Even for Atari, `EADream/envs/wrappers.py` imports `dm_control` at module import
time. Install the pinned `dm-control==1.0.9` and `mujoco==2.3.5` versions above;
do not let pip install the latest `dm_control`.

If pip attempts to build a latest `mujoco` and fails with
`MUJOCO_PATH environment variable is not set`, stop that install. It means the
DMC/MemoryMaze dependency stack was being installed without pins. EADream
Atari100K does not need `memory_maze`.

Run EADream Atari100K:

```bash
PYTHON_BIN=/home/dat.tt2/miniconda3/envs/eadream/bin/python \
GAME=breakout \
SEED=0 \
DEVICE=cuda:0 \
RUN_NAME=eadream-atari100k-breakout-seed0 \
examples/hierarchical_dreamer/baselines/run_eawm_eadream_atari100k.sh
```

Fast smoke:

```bash
PYTHON_BIN=/home/dat.tt2/miniconda3/envs/eadream/bin/python \
GAME=breakout \
SEED=0 \
DEVICE=cuda:0 \
RUN_NAME=eadream-smoke-breakout-seed0 \
STEPS=1000 \
EVAL_EVERY=500 \
LOG_EVERY=500 \
EVAL_EPISODE_NUM=1 \
COMPILE=False \
VIDEO_PRED_LOG=False \
examples/hierarchical_dreamer/baselines/run_eawm_eadream_atari100k.sh
```

EADream logs to TensorBoard files under
`logs/external_baselines/eawm/eadream/<RUN_NAME>/`. It does not have native W&B
logging in the Dreamer script.

## EASimulus Environment

```bash
conda create -n easimulus python=3.10 -y
conda activate easimulus
python -m pip install --upgrade pip setuptools wheel

conda install -y pytorch=2.5.1 torchvision pytorch-cuda=12.4 "mkl<2025" -c pytorch -c nvidia

cd /mnt/disk1/backup_user/dat.tt2/xuance/external_baselines/EAWM/EASimulus
pip install -r requirements.txt
python get_lpips.py
```

Smoke import:

```bash
cd /mnt/disk1/backup_user/dat.tt2/xuance/external_baselines/EAWM/EASimulus
python - <<'PY'
import torch, gymnasium, ale_py, hydra, cv2, wandb
import src.main
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("gymnasium", gymnasium.__version__)
print("EASimulus import OK")
PY
```

Run EASimulus EAWM Atari100K:

```bash
PYTHON_BIN=/home/dat.tt2/miniconda3/envs/easimulus/bin/python \
GAME=Breakout \
SEED=0 \
DEVICE=cuda:0 \
RUN_NAME=easimulus-eawm-atari100k-breakout-seed0 \
WANDB_MODE=online \
PROJECT_NAME=HTS-WM-Baselines \
examples/hierarchical_dreamer/baselines/run_eawm_easimulus_atari100k.sh
```

Fast smoke:

```bash
PYTHON_BIN=/home/dat.tt2/miniconda3/envs/easimulus/bin/python \
GAME=Breakout \
SEED=0 \
DEVICE=cuda:0 \
RUN_NAME=easimulus-eawm-smoke-breakout-seed0 \
WANDB_MODE=disabled \
EPOCHS=1 \
TRAIN_STEPS_PER_EPOCH=10 \
EVAL_EVERY=1 \
WITH_LPIPS=False \
examples/hierarchical_dreamer/baselines/run_eawm_easimulus_atari100k.sh
```

## Protocol Notes

- Local patch: `EADream/envs/wrappers.py` was patched to lazy-import
  `dm_control` and `mjbindings`. Upstream imports them at module load time, which
  makes Atari-only runs fail on servers without a working EGL/OSMesa renderer
  even though Atari does not use DMC wrappers.
- EADream's Atari100K config uses `steps=4e5`, `action_repeat=4`, `envs=1`,
  `batch_size=16`, `batch_length=64`, and `train_ratio=1024`.
- EADream's Atari wrapper uses grayscale 64x64 images, action repeat 4, no-op
  reset, `actions=needed`, `sticky=False`, and `lives=unused` for Atari100K.
- EASimulus's Atari wrapper uses `NoFrameskip-v4`, frame skip 4, train reward
  clipping/life-loss termination, and separate eval settings without reward
  clipping/life-loss termination.
- Treat EAWM as an external-code method baseline, not as a same-code XuanCe
  ablation.
