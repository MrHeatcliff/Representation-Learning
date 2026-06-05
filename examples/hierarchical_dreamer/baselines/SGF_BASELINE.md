# SGF Baseline

This baseline uses the official SGF repository as an external baseline rather
than a guessed reimplementation inside XuanCe.

## Upstream

- Repository: `https://github.com/jrobine/sgf.git`
- Local clone: `external_baselines/sgf`
- Inspected commit: `11fb493b74f3b06ebaba8d71f302574967151568`
- License: MIT

## Current Assessment

Runnable path: `DONE` for Atari100K Breakout seed 1, `PARTIAL EXTERNAL` for
paper-level reporting until more seeds/games are complete.

SGF is a compact official PyTorch Atari codebase with W&B logging and a default
Atari100K config. It is run directly as an external-code baseline rather than
ported into XuanCe.

Completed Breakout seed 1 result:

- Final eval score: `41.53 +- 45.06`
- Final eval episodes: `100`
- W&B run id: `kcwh4nz5`
- Local summary:
  `external_baselines/sgf/wandb/run-20260604_152108-kcwh4nz5/files/wandb-summary.json`

The wrapper has also been updated so future SGF runs use `RUN_NAME` as the W&B
display name via `WANDB_NAME`.

## Atari100K Config Found Upstream

`configs/default.yaml` contains:

- `trainer.env_steps: 100000`
- `trainer.init_steps: 5000`
- `trainer.eval_every: 2500`
- RGB observations: `grayscale: false`
- `resolution: 64`
- `frame_skip: 4`
- `frame_stack: 4`
- `sticky: false`
- `full_action_space: false`
- `noop_max: 30`
- training env `episodic_life: true`
- eval env `episodic_life: false`
- eval episodes: `20`
- final eval episodes: `100`

This is an external-code Atari100K protocol, not a same-code XuanCe ablation.

## Setup In Existing `harmonydream` Env

The existing `harmonydream` env already has PyTorch 2.5.1, Gymnasium,
`ruamel.yaml`, and NumPy. Add the missing packages:

```bash
conda activate harmonydream

pip install wandb torchvision ale-py moviepy

# If cv2.resize() fails despite receiving a numpy array, pin NumPy and OpenCV
# together. OpenCV 4.8.x is compiled against NumPy 1.x and will not import with
# NumPy 2.x.
pip install --force-reinstall "numpy==1.26.4" "opencv-python==4.8.1.78"
```

Check:

```bash
PYTHONPATH=/mnt/disk1/backup_user/dat.tt2/xuance/external_baselines/sgf/src \
python - <<'PY'
import torch, torchvision, gymnasium, ale_py, wandb, ruamel.yaml
import main
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("SGF import OK")
PY
```

Atari check:

```bash
PYTHONPATH=/mnt/disk1/backup_user/dat.tt2/xuance/external_baselines/sgf/src \
python - <<'PY'
import gymnasium as gym
import ale_py
env = gym.make("ale_py:BreakoutNoFrameskip-v4", full_action_space=False)
print(env.observation_space, env.action_space)
env.close()
PY
```

## Smoke Run

```bash
cd /mnt/disk1/backup_user/dat.tt2/xuance
conda activate harmonydream

GAME=Breakout \
SEED=1 \
DEVICE=cuda:0 \
RUN_NAME=sgf-smoke-breakout-seed1 \
WANDB_MODE=disabled \
ENV_STEPS=1000 \
INIT_STEPS=100 \
EVAL_EVERY=500 \
EVAL_EPISODES=1 \
FINAL_EVAL_EPISODES=1 \
WM_BATCH_SIZE=32 \
AGENT_BATCH_SIZE=32 \
AGENT_EVAL=final \
WM_EVAL=none \
AMP=False \
COMPILE=False \
PYTHON_BIN=/home/dat.tt2/miniconda3/envs/harmonydream/bin/python \
examples/hierarchical_dreamer/baselines/run_sgf_atari100k.sh
```

## Full Atari100K Run

```bash
cd /mnt/disk1/backup_user/dat.tt2/xuance
conda activate harmonydream

GAME=Breakout \
SEED=1 \
DEVICE=cuda:0 \
RUN_NAME=sgf-atari100k-breakout-seed1 \
PROJECT_NAME=HTS-WM-Baselines \
WANDB_MODE=online \
ENV_STEPS=100000 \
INIT_STEPS=5000 \
EVAL_EVERY=2500 \
EVAL_EPISODES=20 \
FINAL_EVAL_EPISODES=100 \
WM_BATCH_SIZE=1024 \
AGENT_BATCH_SIZE=3072 \
AGENT_EVAL=all \
WM_EVAL=none \
AMP=True \
COMPILE=False \
PYTHON_BIN=/home/dat.tt2/miniconda3/envs/harmonydream/bin/python \
examples/hierarchical_dreamer/baselines/run_sgf_atari100k.sh
```

The wrapper writes a generated config to:

```text
logs/external_baselines/sgf/<RUN_NAME>/config.yaml
```

SGF logs directly to W&B. With `WANDB_MODE=disabled`, use terminal output for
smoke validation.

## Caveats

- SGF's default training env uses `episodic_life: true`; eval uses
  `episodic_life: false`.
- SGF frame-stacks 4 frames. This differs from current XuanCe HTS-WM Atari
  wrappers.
- SGF is a method-level external baseline, not a same-code ablation.
