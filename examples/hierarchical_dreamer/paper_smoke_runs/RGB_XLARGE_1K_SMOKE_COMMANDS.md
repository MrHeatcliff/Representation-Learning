# RGB Small 1K Smoke Commands

Run this before launching RGB/small paper-final Atari sweeps. It checks that
all same-code conditions use the same preprocessing/model setup:

```text
obs_type = rgb
img_size = 64x64
num_stack = 1
frame_skip / action_repeat = 4
model_size = small
replay_ratio = 0.125
running_steps = 1000
```

## Run Full Smoke Queue

```bash
bash <<'BASH'
set -euo pipefail
cd /mnt/disk1/backup_user/dat.tt2/xuance

DEVICE="${DEVICE:-cuda:0}" \
WANDB_MODE=offline \
PROJECT_NAME=HTS-WM-RGB-Small-Smoke \
ENV_ID=ALE/Breakout-v5 \
SEED=1 \
RUNNING_STEPS=1000 \
START_TRAINING=100 \
EVAL_INTERVAL=500 \
REPLAY_RATIO=1.0 \
BATCH_SIZE=16 \
SEQ_LEN=64 \
OBS_TYPE=rgb \
NUM_STACK=1 \
FRAME_SKIP=4 \
IMG_SIZE_0=64 \
IMG_SIZE_1=64 \
MODEL_SIZE=small \
TEST_EPISODE=1 \
INTERMEDIATE_TEST_EPISODE=1 \
RENDER_EVAL_VIDEO=false \
RENDER_INTERMEDIATE_VIDEO=false \
examples/hierarchical_dreamer/paper_smoke_runs/run_rgb_small_1k_smoke_queue.sh
BASH
```

Use `WANDB_MODE=online` in the command above if you want the smoke runs pushed
to W&B.

To rerun only selected conditions after a fix, add for example:

```bash
SMOKE_METHODS=hts-full,flat-mh,xuance-harmony \
examples/hierarchical_dreamer/paper_smoke_runs/run_rgb_small_1k_smoke_queue.sh
```

## Check Smoke Logs

```bash
cd /mnt/disk1/backup_user/dat.tt2/xuance

rg -n "Traceback|RuntimeError|Exception|CUDA out|out of memory|nan|NaN" \
  logs/training_scripts/rgb_small_1k_smoke_* -S

find logs/rgb_small_smoke -path "*eval_results/final_eval.json" -print
```

## Expected Coverage

This queue covers:

- DreamerV3 same-code baseline
- HTS-WM full
- Flat-SAE
- Flat-MH
- SGF-style flat same-code
- Recon-only hierarchy
- Dense multi-stride no-sparse
- HTS no-hier
- HTS no-sdyn
- HTS no-temp
- HTS no-VC
- Larger-flat-param on Breakout
- XuanCe HarmonyDream same-code
- TSAE-style same-code

## Current Smoke Status

Last checked: 2026-06-09.

Setup checked:

```text
env_id = ALE/Breakout-v5
obs_type = rgb
img_size = 64x64
num_stack = 1
frame_skip / action_repeat = 4
model_size = small
replay_ratio = 0.125
running_steps = 1000
start_training = 100
```

| Condition | Status | Notes |
|---|---|---|
| DreamerV3 | PASS | Final eval artifact exists. |
| HTS-WM full | PASS | Final eval artifact exists. Peak allocated CUDA memory about 13.4GB in W&B console summary. |
| Flat-SAE | PASS | Final eval artifact exists. |
| Flat-MH | PASS | Final eval artifact exists. |
| SGF-style flat same-code | PARTIAL | Training reached 1000 steps, then smoke was stopped before final eval to avoid sharing `cuda:0` with an active full Atari run. |
| Recon-only hierarchy | NOT RUN | Resume command below. |
| Dense multi-stride no-sparse | NOT RUN | Resume command below. |
| HTS no-hier | NOT RUN | Resume command below. |
| HTS no-sdyn | NOT RUN | Resume command below. |
| HTS no-temp | NOT RUN | Resume command below. |
| HTS no-VC | NOT RUN | Resume command below. |
| Larger-flat-param | NOT RUN | Resume command below, Breakout-specific. |
| XuanCe HarmonyDream same-code | NOT RUN | Resume command below. |
| TSAE-style same-code | NOT RUN | Resume command below. |

Resume the remaining checks on a free GPU:

```bash
bash <<'BASH'
set -euo pipefail
cd /mnt/disk1/backup_user/dat.tt2/xuance

DEVICE="${DEVICE:-cuda:1}" \
WANDB_MODE=offline \
PROJECT_NAME=HTS-WM-RGB-Small-Smoke \
ENV_ID=ALE/Breakout-v5 \
SEED=1 \
RUNNING_STEPS=1000 \
START_TRAINING=100 \
EVAL_INTERVAL=500 \
REPLAY_RATIO=1.0 \
BATCH_SIZE=16 \
SEQ_LEN=64 \
OBS_TYPE=rgb \
NUM_STACK=1 \
FRAME_SKIP=4 \
IMG_SIZE_0=64 \
IMG_SIZE_1=64 \
MODEL_SIZE=small \
TEST_EPISODE=1 \
INTERMEDIATE_TEST_EPISODE=1 \
RENDER_EVAL_VIDEO=false \
RENDER_INTERMEDIATE_VIDEO=false \
SMOKE_METHODS=sgf-style-flat-same-code,recon-only-hierarchy,dense-multistride-no-sparse,hts-no-hier,hts-no-sdyn,hts-no-temp,hts-no-vc,larger-flat-param,xuance-harmony,tsae-style \
examples/hierarchical_dreamer/paper_smoke_runs/run_rgb_small_1k_smoke_queue.sh
BASH
```
