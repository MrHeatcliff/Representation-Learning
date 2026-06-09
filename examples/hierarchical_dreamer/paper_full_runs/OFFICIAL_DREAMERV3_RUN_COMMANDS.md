# Official DreamerV3 Atari100K Run Commands

Repo cloned at:

```bash
/mnt/disk1/backup_user/dat.tt2/xuance/external_baselines/dreamerv3-official
```

Current cloned commit:

```text
e3f02248693a79dc8b0ebd62c93683888ddaccfe
```

Verified config facts from `dreamerv3/configs.yaml`:

```text
atari100k task default: atari100k_pong
run.steps: 1.1e5
run.envs: 1
run.train_ratio: 256
batch_size: 16
batch_length: 64
env.atari100k.size: [64, 64]
env.atari100k.repeat: 4
env.atari100k.sticky: false
env.atari100k.gray: false
env.atari100k.actions: needed
env.atari100k.lives: unused
env.atari100k.clip_reward: false
```

## Setup Separate Env

Do this in a separate env because the current HarmonyDream env does not have `elements`, and official DreamerV3 pins `jax[cuda12]==0.4.33`.

```bash
cd /mnt/disk1/backup_user/dat.tt2/xuance/external_baselines/dreamerv3-official

conda create -y -n dreamerv3official python=3.11
conda activate dreamerv3official

python -m pip install -U pip setuptools wheel
python -m pip install -U -r requirements.txt

AutoROM --accept-license

python - <<'PY'
import jax, ale_py, elements, portal, ninjax
import dreamerv3.main
print("jax", jax.__version__)
print("devices", jax.devices())
print("ale_py", ale_py.__version__)
print("official DreamerV3 import OK")
PY
```

## Alien 10K Smoke

This uses the current official `atari100k` config and only overrides `run.steps` to make a short smoke run. Do not use this for paper results.

```bash
cd /mnt/disk1/backup_user/dat.tt2/xuance/external_baselines/dreamerv3-official
conda activate dreamerv3official

CUDA_VISIBLE_DEVICES=0 \
WANDB_PROJECT=HTS-WM-HarmonyDream-Alien-Curve \
WANDB_RUN_NAME=DreamerV3-official-atari100k-alien-smoke10k-seed0 \
WANDB_MODE=online \
WANDB_GROUP=official-dreamerv3-atari100k-smoke \
WANDB_TAGS=official-dreamerv3,atari100k,alien,smoke,seed0 \
python dreamerv3/main.py \
  --configs atari100k \
  --task atari100k_alien \
  --seed 0 \
  --logdir /mnt/disk1/backup_user/dat.tt2/xuance/logs/external_baselines/dreamerv3_official/alien_smoke_10k_current_main_seed0 \
  --run.steps 10000 \
  --logger.outputs jsonl,scope,wandb
```

Read scores:

```bash
tail -n 20 /mnt/disk1/backup_user/dat.tt2/xuance/logs/external_baselines/dreamerv3_official/alien_smoke_10k_current_main_seed0/scores.jsonl
tail -n 20 /mnt/disk1/backup_user/dat.tt2/xuance/logs/external_baselines/dreamerv3_official/alien_smoke_10k_current_main_seed0/metrics.jsonl
```

## Alien Atari100K, Exact Current Official Config

This leaves the Atari100K training setup exactly as defined by the cloned official repo:

```text
--configs atari100k
run.steps = 1.1e5
run.envs = 1
run.train_ratio = 256
batch_size = 16
batch_length = 64
env.atari100k.repeat = 4
env.atari100k.sticky = false
env.atari100k.gray = false
env.atari100k.clip_reward = false
```

The only extra flag below is `--logger.outputs jsonl,scope,wandb`, which enables WandB logging. It does not change training.

```bash
cd /mnt/disk1/backup_user/dat.tt2/xuance/external_baselines/dreamerv3-official
conda activate dreamerv3official

CUDA_VISIBLE_DEVICES=0 \
WANDB_PROJECT=HTS-WM-HarmonyDream-Alien-Curve \
WANDB_RUN_NAME=DreamerV3-official-atari100k-alien-seed0 \
WANDB_MODE=online \
WANDB_GROUP=official-dreamerv3-atari100k \
WANDB_TAGS=official-dreamerv3,atari100k,alien,seed0 \
python dreamerv3/main.py \
  --configs atari100k \
  --task atari100k_alien \
  --seed 0 \
  --logdir /mnt/disk1/backup_user/dat.tt2/xuance/logs/external_baselines/dreamerv3_official/alien_atari100k_official_seed0 \
  --logger.outputs jsonl,scope,wandb
```

## Alien Atari100K, Official Small Preset

The current official repo does not have a config named `small`. It provides model-size presets:

```text
size1m, size12m, size25m, size50m, size100m, size200m, size400m
```

The default architecture in `defaults` is effectively the `size200m` setting. To test a small official model while keeping the Atari100K training protocol from the repo, add `size12m` after `atari100k`:

```bash
cd /mnt/disk1/backup_user/dat.tt2/xuance/external_baselines/dreamerv3-official
conda activate dreamerv3official

CUDA_VISIBLE_DEVICES=0 \
WANDB_PROJECT=HTS-WM-HarmonyDream-Alien-Curve \
WANDB_RUN_NAME=DreamerV3-official-size12m-atari100k-alien-seed0 \
WANDB_MODE=online \
WANDB_GROUP=official-dreamerv3-atari100k-size12m \
WANDB_TAGS=official-dreamerv3,atari100k,alien,size12m,seed0 \
python dreamerv3/main.py \
  --configs atari100k size12m \
  --task atari100k_alien \
  --seed 0 \
  --logdir /mnt/disk1/backup_user/dat.tt2/xuance/logs/external_baselines/dreamerv3_official/alien_atari100k_official_size12m_seed0 \
  --logger.outputs jsonl,scope,wandb
```

For a short smoke test of the same small official model:

```bash
cd /mnt/disk1/backup_user/dat.tt2/xuance/external_baselines/dreamerv3-official
conda activate dreamerv3official

CUDA_VISIBLE_DEVICES=0 \
WANDB_PROJECT=HTS-WM-HarmonyDream-Alien-Curve \
WANDB_RUN_NAME=DreamerV3-official-size12m-atari100k-alien-smoke10k-seed0 \
WANDB_MODE=online \
WANDB_GROUP=official-dreamerv3-atari100k-size12m-smoke \
WANDB_TAGS=official-dreamerv3,atari100k,alien,size12m,smoke,seed0 \
python dreamerv3/main.py \
  --configs atari100k size12m \
  --task atari100k_alien \
  --seed 0 \
  --logdir /mnt/disk1/backup_user/dat.tt2/xuance/logs/external_baselines/dreamerv3_official/alien_smoke_10k_official_size12m_seed0 \
  --run.steps 10000 \
  --logger.outputs jsonl,scope,wandb
```

## Breakout Atari100K, Exact Current Official Config

Use this if you want to compare against the XuanCe Breakout runs first.

```bash
cd /mnt/disk1/backup_user/dat.tt2/xuance/external_baselines/dreamerv3-official
conda activate dreamerv3official

CUDA_VISIBLE_DEVICES=0 \
WANDB_PROJECT=HTS-WM-HarmonyDream-Alien-Curve \
WANDB_RUN_NAME=DreamerV3-official-atari100k-breakout-seed0 \
WANDB_MODE=online \
WANDB_GROUP=official-dreamerv3-atari100k \
WANDB_TAGS=official-dreamerv3,atari100k,breakout,seed0 \
python dreamerv3/main.py \
  --configs atari100k \
  --task atari100k_breakout \
  --seed 0 \
  --logdir /mnt/disk1/backup_user/dat.tt2/xuance/logs/external_baselines/dreamerv3_official/breakout_atari100k_official_seed0 \
  --logger.outputs jsonl,scope,wandb
```

## Optional WandB Output

This clone was patched to let the official `WandBOutput` read these env vars:

```bash
WANDB_PROJECT
WANDB_RUN_NAME
WANDB_MODE
WANDB_ENTITY
WANDB_GROUP
WANDB_JOB_TYPE
WANDB_TAGS
```

If flag parsing for `--logger.outputs jsonl,scope,wandb` fails in your env, rerun without WandB and keep the JSONL files:

```bash
--logger.outputs jsonl,scope
```
