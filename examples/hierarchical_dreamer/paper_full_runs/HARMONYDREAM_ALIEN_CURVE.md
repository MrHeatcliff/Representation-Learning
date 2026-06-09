# HarmonyDream Alien Curve Probe

Use this before launching full Atari100K sweeps. It follows the HarmonyDream Atari100K reproduction target:

```text
model_size = size12m
obs = RGB 64x64, stack 1
action repeat = 4
sticky action = 0.0
episodic life = false
reward clipping = false
replay_ratio = 0.25
batch = 16 x 64
eval = 100 episodes every 5000 policy steps
```

## 1. Smoke 10K

Run DreamerV3 reproduced first. This should verify protocol metrics and parameter count before the expensive 100K run.

```bash
cd /mnt/disk1/backup_user/dat.tt2/xuance

METHODS=dreamer \
SEEDS=0 \
ENV_ID=ALE/Alien-v5 \
DEVICE=cuda:0 \
HARMONY_MODEL_SIZE=size12m \
WANDB_MODE=online \
PROJECT_NAME=HTS-WM-HarmonyDream-Alien-Curve \
RUNNING_STEPS=10000 \
EVAL_INTERVAL=5000 \
TEST_EPISODE=20 \
INTERMEDIATE_TEST_EPISODE=20 \
RENDER_EVAL_VIDEO=false \
RENDER_INTERMEDIATE_VIDEO=false \
examples/hierarchical_dreamer/paper_full_runs/run_harmonydream_alien_curve.sh
```

Expected invariants near the end:

```text
debug/sticky_action_probability = 0.0
debug/clip_reward = 0
debug/episodic_life = 0
debug/env_frame = debug/agent_step * 4
debug/gradient_step ~= 0.25 * (agent_step - 1024)
params/core_agent_total should be checked against the official size12m-scale run
```

## 2. Alien 100K, DreamerV3 Reproduced

```bash
cd /mnt/disk1/backup_user/dat.tt2/xuance

METHODS=dreamer \
SEEDS=0 \
ENV_ID=ALE/Alien-v5 \
DEVICE=cuda:0 \
HARMONY_MODEL_SIZE=size12m \
WANDB_MODE=online \
PROJECT_NAME=HTS-WM-HarmonyDream-Alien-Curve \
RUNNING_STEPS=100000 \
EVAL_INTERVAL=5000 \
TEST_EPISODE=100 \
INTERMEDIATE_TEST_EPISODE=100 \
RENDER_EVAL_VIDEO=true \
RENDER_INTERMEDIATE_VIDEO=false \
examples/hierarchical_dreamer/paper_full_runs/run_harmonydream_alien_curve.sh
```

## 3. Alien 100K, HarmonyDream

Run this only after the DreamerV3 reproduced curve looks sane.

```bash
cd /mnt/disk1/backup_user/dat.tt2/xuance

METHODS=harmony \
SEEDS=0 \
ENV_ID=ALE/Alien-v5 \
DEVICE=cuda:0 \
HARMONY_MODEL_SIZE=size12m \
WANDB_MODE=online \
PROJECT_NAME=HTS-WM-HarmonyDream-Alien-Curve \
RUNNING_STEPS=100000 \
EVAL_INTERVAL=5000 \
TEST_EPISODE=100 \
INTERMEDIATE_TEST_EPISODE=100 \
RENDER_EVAL_VIDEO=true \
RENDER_INTERMEDIATE_VIDEO=false \
examples/hierarchical_dreamer/paper_full_runs/run_harmonydream_alien_curve.sh
```

## 4. Five Seeds After Alien Seed 0 Looks Good

```bash
cd /mnt/disk1/backup_user/dat.tt2/xuance

METHODS=dreamer,harmony \
SEEDS=0,1,2,3,4 \
ENV_ID=ALE/Alien-v5 \
DEVICE=cuda:0 \
HARMONY_MODEL_SIZE=size12m \
WANDB_MODE=online \
PROJECT_NAME=HTS-WM-HarmonyDream-Alien-Curve \
RUNNING_STEPS=100000 \
EVAL_INTERVAL=5000 \
TEST_EPISODE=100 \
INTERMEDIATE_TEST_EPISODE=100 \
RENDER_EVAL_VIDEO=true \
RENDER_INTERMEDIATE_VIDEO=false \
examples/hierarchical_dreamer/paper_full_runs/run_harmonydream_alien_curve.sh
```
