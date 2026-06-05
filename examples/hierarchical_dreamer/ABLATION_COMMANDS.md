# HTS-WM Paper Experiment Command Templates

This file maps every current `paper.txt` table and figure to runnable configs,
known gaps, and command templates. Use the labels from the paper as stable IDs
for W&B run names, generated config names, artifact folders, and plot scripts.

For commands that can be pasted directly into the terminal, use
`examples/hierarchical_dreamer/PASTE_READY_BASELINE_RUNS.md`.
For the updated paper table/figure checklist, use
`examples/hierarchical_dreamer/PAPER_EXPERIMENT_REGISTRY.md`.

Status legend:

- `READY`: can be launched with the current Hierarchical Dreamer code.
- `PARTIAL`: the main training variant can run, but some baselines, metrics, or
  plot extraction are missing.
- `MISSING`: needs new environment, baseline implementation, analysis script, or
  metric logging before it can be produced.

## Start Here: Paste-Ready Breakout Runs

Run one block at a time. These blocks are complete commands with no placeholders.
Smoke runs are short dependency checks; full runs are the commands to fill the
current Atari100K Breakout rows.

Completed Breakout seed 1 runs are tracked in
`examples/hierarchical_dreamer/RESULTS_TRACKER.md`.

The updated paper requires final-checkpoint same-code Atari results with `100`
eval episodes. Current DreamerV3 and HTS-WM runs are development-complete but
not paper-final.

| Method | Status | Score Used In Tracker | W&B Run | Notes |
|---|---|---:|---|---|
| DreamerV3 | DEV_DONE | `15.67 +- 2.05` | `cssk65zq` | Best checkpoint over 3 eval episodes; not paper-final. |
| HTS-WM | DEV_DONE | `15.33 +- 1.70` | `i95tp2se` | Best checkpoint over 3 eval episodes; not paper-final. |
| SGF | PARTIAL EXTERNAL | `41.53 +- 45.06` | `kcwh4nz5` | Official-code final eval over 100 episodes, single seed. |

Before paper-final same-code reruns, update the local launchers/configs for the
paper protocol: final checkpoint only and `100` Atari eval episodes. The current
paste-ready local DreamerV3/HTS-WM commands below remain development commands.

### Local XuanCe Runs

DreamerV3 anchor (DEV_DONE: Breakout seed 1, W&B `cssk65zq`):

```bash
cd /mnt/disk1/backup_user/dat.tt2/xuance

PYTHON_BIN=/mnt/disk1/backup_user/dat.tt2/xuance/.venv/bin/python
"$PYTHON_BIN" examples/dreamer_v3/dreamer_v3_atari.py \
  --env-id ALE/Breakout-v5 \
  --device cuda:0 \
  --logger wandb \
  --project-name HTS-WM-Baselines \
  --wandb-mode online \
  --wandb-run-name DreamerV3-baseline-breakout-seed1-100k \
  --running-steps 100000 \
  --replay-ratio 1 \
  --batch-size 16 \
  --seq-len 64 \
  --benchmark 1
```

Full HTS-WM / Hierarchical Dreamer (DEV_DONE: Breakout seed 1, W&B `i95tp2se`):

```bash
cd /mnt/disk1/backup_user/dat.tt2/xuance

PYTHON_BIN=/mnt/disk1/backup_user/dat.tt2/xuance/.venv/bin/python \
CONFIG_FILE=config/atari100k_two_phase.yaml \
RUN_NAME=HTS-WM-full-breakout-seed1-100k \
ENV_ID=ALE/Breakout-v5 \
DEVICE=cuda:0 \
WANDB_MODE=online \
PROJECT_NAME=HTS-WM-Baselines \
RUNNING_STEPS=100000 \
REPLAY_RATIO=1 \
examples/hierarchical_dreamer/train_ablation.sh
```

T-SAE-style temporal-only control:

```bash
cd /mnt/disk1/backup_user/dat.tt2/xuance

PYTHON_BIN=/mnt/disk1/backup_user/dat.tt2/xuance/.venv/bin/python \
SEED=1 \
DEVICE=cuda:0 \
RUN_NAME=tsae-style-breakout-seed1-100k \
WANDB_MODE=online \
PROJECT_NAME=HTS-WM-Baselines \
RUNNING_STEPS=100000 \
REPLAY_RATIO=1 \
examples/hierarchical_dreamer/baselines/run_tsae_style_atari100k.sh
```

Same-code XuanCe HarmonyDream:

```bash
cd /mnt/disk1/backup_user/dat.tt2/xuance

PYTHON_BIN=/mnt/disk1/backup_user/dat.tt2/xuance/.venv/bin/python \
ENV_ID=ALE/Breakout-v5 \
SEED=1 \
DEVICE=cuda:0 \
REPLAY_RATIO=1 \
RUNNING_STEPS=100000 \
WANDB_MODE=online \
PROJECT_NAME=HTS-WM-Baselines \
RUN_NAME=XuanCe-HarmonyDream-breakout-seed1-100k \
examples/hierarchical_dreamer/baselines/run_xuance_harmonydream_atari100k.sh
```

### DyMoDreamer Official-Code Runs

Smoke:

```bash
cd /mnt/disk1/backup_user/dat.tt2/xuance

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
PYTHON_BIN=/home/dat.tt2/miniconda3/envs/harmonydream/bin/python \
examples/hierarchical_dreamer/baselines/run_dymodreamer_atari100k.sh
```

Full Atari100K (NOT DONE):

```bash
cd /mnt/disk1/backup_user/dat.tt2/xuance

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
PYTHON_BIN=/home/dat.tt2/miniconda3/envs/harmonydream/bin/python \
examples/hierarchical_dreamer/baselines/run_dymodreamer_atari100k.sh
```

### SGF Official-Code Runs

Smoke:

```bash
cd /mnt/disk1/backup_user/dat.tt2/xuance

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
AGENT_EVAL=final \
WM_EVAL=none \
WM_BATCH_SIZE=32 \
AGENT_BATCH_SIZE=32 \
AMP=False \
COMPILE=False \
PYTHON_BIN=/home/dat.tt2/miniconda3/envs/harmonydream/bin/python \
examples/hierarchical_dreamer/baselines/run_sgf_atari100k.sh
```

Full Atari100K (PARTIAL EXTERNAL: Breakout seed 1, W&B `kcwh4nz5`):

```bash
cd /mnt/disk1/backup_user/dat.tt2/xuance

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
AGENT_EVAL=all \
WM_EVAL=none \
AMP=True \
COMPILE=False \
PYTHON_BIN=/home/dat.tt2/miniconda3/envs/harmonydream/bin/python \
examples/hierarchical_dreamer/baselines/run_sgf_atari100k.sh
```

### EAWM / EADream Official-Code Runs

Smoke:

```bash
cd /mnt/disk1/backup_user/dat.tt2/xuance

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

Full Atari100K:

```bash
cd /mnt/disk1/backup_user/dat.tt2/xuance

PYTHON_BIN=/home/dat.tt2/miniconda3/envs/eadream/bin/python \
GAME=breakout \
SEED=0 \
DEVICE=cuda:0 \
RUN_NAME=eadream-atari100k-breakout-seed0 \
STEPS=4e5 \
TRAIN_RATIO=1024 \
EVAL_EVERY=3e4 \
LOG_EVERY=3e4 \
EVAL_EPISODE_NUM=100 \
COMPILE=False \
VIDEO_PRED_LOG=True \
examples/hierarchical_dreamer/baselines/run_eawm_eadream_atari100k.sh
```

## Reference Only: Shared Variables And Config Templates

Do not paste this section directly. It contains reusable variables and template
patterns for generating new ablation configs.

```bash
cd /mnt/disk1/backup_user/dat.tt2/xuance

BASE=examples/hierarchical_dreamer/config/atari100k_two_phase.yaml
OUT=examples/hierarchical_dreamer/config/generated_configs
DEVICE=cuda:0
WANDB_MODE=online
PROJECT_NAME=HTS-WM-Ablations
ENV_ID=ALE/Breakout-v5
RUNNING_STEPS=100000

# XuanCe replay_ratio is minibatch updates per agent step.
# Approximate DreamerV3 current-main train_ratio=256 with 16x64 batches:
REPLAY_RATIO=0.25

# Approximate DreamerV3 paper replay_ratio=128 with 16x64 batches:
# REPLAY_RATIO=0.125
```

Reference generate-and-launch pattern. Replace every placeholder before running:

```bash
.venv/bin/python examples/hierarchical_dreamer/make_ablation_config.py \
  --base "$BASE" \
  --output "$OUT/<experiment_id>.yaml" \
  --set hierarchical_latent.ablation_name=<experiment_id>

CONFIG_FILE=config/generated_configs/<experiment_id>.yaml \
RUN_NAME=<experiment_id>-breakout-seed1 \
ENV_ID=$ENV_ID DEVICE=$DEVICE WANDB_MODE=$WANDB_MODE PROJECT_NAME=$PROJECT_NAME \
RUNNING_STEPS=$RUNNING_STEPS REPLAY_RATIO=$REPLAY_RATIO \
examples/hierarchical_dreamer/train_ablation.sh
```

## Reference Only: Currently Exposed Flags

Do not paste these `--set` lines by themselves. They are arguments for
`make_ablation_config.py`.

```bash
# hierarchy shape
--set hierarchical_latent.num_heads=6
--set hierarchical_latent.head_dim=32
--set hierarchical_latent.sparsity_topk='[8,8,8,8,8,8]'

# nested reconstruction
--set hierarchical_latent.reconstruction.betas='[1,1,1,1,1,1]'
--set hierarchical_latent.reconstruction.stop_gradient_lower_levels=true

# sparse dynamics
--set hierarchical_latent.sparse_dynamics.strides='[32,16,8,4,2,1]'
--set hierarchical_latent.sparse_dynamics.alphas='[1,1,1,1,1,1]'
--set hierarchical_latent.sparse_dynamics.stop_gradient_lower_levels=true
--set hierarchical_latent.sparse_dynamics.require_strict_decreasing=true
--set hierarchical_latent.sparse_dynamics.target_stop_gradient=true
--set hierarchical_latent.sparse_dynamics.action_mode=subsequence

# temporal and collapse control
--set hierarchical_latent.temporal_consistency.mode=contrastive
--set hierarchical_latent.temporal_consistency.projector_type=nonlinear
--set hierarchical_latent.temporal_consistency.projection_dim=64
--set hierarchical_latent.temporal_consistency.far_negative_mode=hard
--set hierarchical_latent.variance_covariance.mode=both

# sparsity
--set hierarchical_latent.sparsity.mode=level_topk
--set hierarchical_latent.sparsity.gammas='[0.0001,0.0001,0.0001,0.0001,0.0001,0.0001]'

# loss weights
--set hierarchical_latent.loss_weights.hierarchical=1.0
--set hierarchical_latent.loss_weights.sparse_dynamics=1.0
--set hierarchical_latent.loss_weights.temporal=1.0
--set hierarchical_latent.loss_weights.variance_covariance=1.0
--set hierarchical_latent.loss_weights.sparsity=1.0

# training regime
--set training_regime.name=two_phase
--set training_regime.freeze_encoder=false
--set training_regime.wm_only_phase1=true
--set training_regime.phase1_gradient_steps=20000
--set training_regime.encoder_lr_scale=0.1
--set training_regime.hierarchy_lr_scale=1.0

# backbone knobs already present in config
--set world_model.kl_dynamic=0.5
--set world_model.kl_representation=0.1
--set world_model.kl_free_nats=1.0
--set world_model.recurrent_model.recurrent_state_size=512
--set world_model.recurrent_model.dense_units=512
--set world_model.discrete_size=32
--set world_model.stochastic_size=32
```

## Baseline Commands

### Immediate Run Queue For Current Baselines

Use this section first if the goal is to start filling the current paper tables
with the baselines that already have local wrappers. These commands target
Atari100K Breakout as the first reproducibility guardrail.

#### Run Group: `tab:baselines` / `tab:main-results`

Local XuanCe baselines use the repo `.venv`:

```bash
cd /mnt/disk1/backup_user/dat.tt2/xuance
export PYTHON_BIN=/mnt/disk1/backup_user/dat.tt2/xuance/.venv/bin/python
export ENV_ID=ALE/Breakout-v5
export DEVICE=cuda:0
export WANDB_MODE=online
export PROJECT_NAME=HTS-WM-Baselines
export RUNNING_STEPS=100000
export REPLAY_RATIO=1
```

Run DreamerV3 anchor (DEV_DONE: Breakout seed 1, W&B `cssk65zq`):

```bash
"$PYTHON_BIN" examples/dreamer_v3/dreamer_v3_atari.py \
  --env-id "$ENV_ID" \
  --device "$DEVICE" \
  --logger wandb \
  --project-name "$PROJECT_NAME" \
  --wandb-mode "$WANDB_MODE" \
  --wandb-run-name DreamerV3-baseline-breakout-seed1-100k \
  --running-steps "$RUNNING_STEPS" \
  --replay-ratio "$REPLAY_RATIO" \
  --batch-size 16 \
  --seq-len 64 \
  --benchmark 1
```

Run full HTS-WM (DEV_DONE: Breakout seed 1, W&B `i95tp2se`):

```bash
CONFIG_FILE=config/atari100k_two_phase.yaml \
RUN_NAME=HTS-WM-full-breakout-seed1-100k \
ENV_ID=$ENV_ID DEVICE=$DEVICE WANDB_MODE=$WANDB_MODE PROJECT_NAME=$PROJECT_NAME \
RUNNING_STEPS=$RUNNING_STEPS REPLAY_RATIO=$REPLAY_RATIO PYTHON_BIN=$PYTHON_BIN \
examples/hierarchical_dreamer/train_ablation.sh
```

Run T-SAE-style temporal-only control:

```bash
PYTHON_BIN=$PYTHON_BIN \
SEED=1 \
DEVICE=$DEVICE \
RUN_NAME=tsae-style-breakout-seed1-100k \
WANDB_MODE=$WANDB_MODE \
PROJECT_NAME=$PROJECT_NAME \
RUNNING_STEPS=$RUNNING_STEPS \
REPLAY_RATIO=$REPLAY_RATIO \
examples/hierarchical_dreamer/baselines/run_tsae_style_atari100k.sh
```

Run same-code XuanCe HarmonyDream:

```bash
PYTHON_BIN=$PYTHON_BIN \
ENV_ID=$ENV_ID \
SEED=1 \
DEVICE=$DEVICE \
REPLAY_RATIO=$REPLAY_RATIO \
RUNNING_STEPS=$RUNNING_STEPS \
WANDB_MODE=$WANDB_MODE \
PROJECT_NAME=$PROJECT_NAME \
RUN_NAME=XuanCe-HarmonyDream-breakout-seed1-100k \
examples/hierarchical_dreamer/baselines/run_xuance_harmonydream_atari100k.sh
```

#### Run Group: External Official-Code Baselines

These commands use the existing `harmonydream` conda env that has already been
used for DyMoDreamer and SGF import checks.

DyMoDreamer smoke:

```bash
cd /mnt/disk1/backup_user/dat.tt2/xuance

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
PYTHON_BIN=/home/dat.tt2/miniconda3/envs/harmonydream/bin/python \
examples/hierarchical_dreamer/baselines/run_dymodreamer_atari100k.sh
```

DyMoDreamer full Atari100K (NOT DONE):

```bash
cd /mnt/disk1/backup_user/dat.tt2/xuance

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
PYTHON_BIN=/home/dat.tt2/miniconda3/envs/harmonydream/bin/python \
examples/hierarchical_dreamer/baselines/run_dymodreamer_atari100k.sh
```

SGF smoke. Keep small batch sizes for this smoke because the replay buffer is
tiny at 1000 steps:

```bash
cd /mnt/disk1/backup_user/dat.tt2/xuance

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
AGENT_EVAL=final \
WM_EVAL=none \
WM_BATCH_SIZE=32 \
AGENT_BATCH_SIZE=32 \
AMP=False \
COMPILE=False \
PYTHON_BIN=/home/dat.tt2/miniconda3/envs/harmonydream/bin/python \
examples/hierarchical_dreamer/baselines/run_sgf_atari100k.sh
```

SGF full Atari100K (PARTIAL EXTERNAL: Breakout seed 1, W&B `kcwh4nz5`):

```bash
cd /mnt/disk1/backup_user/dat.tt2/xuance

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
AGENT_EVAL=all \
WM_EVAL=none \
AMP=True \
COMPILE=False \
PYTHON_BIN=/home/dat.tt2/miniconda3/envs/harmonydream/bin/python \
examples/hierarchical_dreamer/baselines/run_sgf_atari100k.sh
```

EAWM EADream smoke:

```bash
cd /mnt/disk1/backup_user/dat.tt2/xuance

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

EAWM EADream full Atari100K:

```bash
cd /mnt/disk1/backup_user/dat.tt2/xuance

PYTHON_BIN=/home/dat.tt2/miniconda3/envs/eadream/bin/python \
GAME=breakout \
SEED=0 \
DEVICE=cuda:0 \
RUN_NAME=eadream-atari100k-breakout-seed0 \
STEPS=4e5 \
TRAIN_RATIO=1024 \
EVAL_EVERY=3e4 \
LOG_EVERY=3e4 \
EVAL_EPISODE_NUM=100 \
COMPILE=False \
VIDEO_PRED_LOG=True \
examples/hierarchical_dreamer/baselines/run_eawm_eadream_atari100k.sh
```

#### Deferred Baselines

EASimulus EAWM is still deferred until the `easimulus` env is installed and
smoke-tested:

- `examples/hierarchical_dreamer/baselines/run_eawm_easimulus_atari100k.sh`

Use EADream above as the current EAWM baseline path.

### Dreamer Backbone

`READY` for the local XuanCe DreamerV3 baseline.

```bash
.venv/bin/python examples/dreamer_v3/dreamer_v3_atari.py \
  --env-id "$ENV_ID" \
  --device "$DEVICE" \
  --logger wandb \
  --project-name "$PROJECT_NAME" \
  --wandb-mode "$WANDB_MODE" \
  --wandb-run-name DreamerV3-baseline-breakout-seed1 \
  --running-steps "$RUNNING_STEPS" \
  --replay-ratio "$REPLAY_RATIO" \
  --batch-size 16 \
  --seq-len 64 \
  --benchmark 1
```

### Full HTS-WM

`READY`.

```bash
.venv/bin/python examples/hierarchical_dreamer/make_ablation_config.py \
  --base "$BASE" \
  --output "$OUT/full_htswm.yaml" \
  --set hierarchical_latent.ablation_name=full_htswm

CONFIG_FILE=config/generated_configs/full_htswm.yaml \
RUN_NAME=full-htswm-breakout-seed1 \
ENV_ID=$ENV_ID DEVICE=$DEVICE WANDB_MODE=$WANDB_MODE PROJECT_NAME=$PROJECT_NAME \
RUNNING_STEPS=$RUNNING_STEPS REPLAY_RATIO=$REPLAY_RATIO \
examples/hierarchical_dreamer/train_ablation.sh
```

### Matryoshka-Only Port

`READY`: nested reconstruction on; sparse dynamics, temporal, and VC off.

```bash
.venv/bin/python examples/hierarchical_dreamer/make_ablation_config.py \
  --base "$BASE" \
  --output "$OUT/matryoshka_only.yaml" \
  --set hierarchical_latent.ablation_name=matryoshka_only \
  --set hierarchical_latent.loss_weights.sparse_dynamics=0.0 \
  --set hierarchical_latent.loss_weights.temporal=0.0 \
  --set hierarchical_latent.loss_weights.variance_covariance=0.0 \
  --set hierarchical_latent.temporal_consistency.mode=none \
  --set hierarchical_latent.variance_covariance.mode=none
```

### Static Recon-Only Hierarchy

`READY`: same as Matryoshka-only but keeps optional VC/sparsity as architecture
diagnostics.

```bash
.venv/bin/python examples/hierarchical_dreamer/make_ablation_config.py \
  --base "$BASE" \
  --output "$OUT/recon_only_hierarchy.yaml" \
  --set hierarchical_latent.ablation_name=recon_only_hierarchy \
  --set hierarchical_latent.loss_weights.sparse_dynamics=0.0 \
  --set hierarchical_latent.loss_weights.temporal=0.0 \
  --set hierarchical_latent.temporal_consistency.mode=none
```

### HTS-WM Without Individual Losses

`READY`.

```bash
# no temporal regularizer
--set hierarchical_latent.ablation_name=no_l_temp \
--set hierarchical_latent.loss_weights.temporal=0.0 \
--set hierarchical_latent.temporal_consistency.mode=none

# no variance-covariance regularizer
--set hierarchical_latent.ablation_name=no_l_vc \
--set hierarchical_latent.loss_weights.variance_covariance=0.0 \
--set hierarchical_latent.variance_covariance.mode=none

# no sparse dynamics
--set hierarchical_latent.ablation_name=no_l_sdyn \
--set hierarchical_latent.loss_weights.sparse_dynamics=0.0

# no nested reconstruction
--set hierarchical_latent.ablation_name=no_l_hier \
--set hierarchical_latent.loss_weights.hierarchical=0.0
```

### Flat Single-Level SAE

`READY` as a sparsity-only approximation with `L=1`.

```bash
.venv/bin/python examples/hierarchical_dreamer/make_ablation_config.py \
  --base "$BASE" \
  --output "$OUT/flat_single_level_sae.yaml" \
  --set hierarchical_latent.ablation_name=flat_single_level_sae \
  --set hierarchical_latent.num_heads=1 \
  --set hierarchical_latent.sparsity_topk='[48]' \
  --set hierarchical_latent.reconstruction.betas='[1.0]' \
  --set hierarchical_latent.sparse_dynamics.strides='[1]' \
  --set hierarchical_latent.sparse_dynamics.alphas='[1.0]' \
  --set hierarchical_latent.loss_weights.sparse_dynamics=0.0 \
  --set hierarchical_latent.loss_weights.temporal=0.0 \
  --set hierarchical_latent.loss_weights.variance_covariance=0.0
```

### Flat Multi-Horizon Control

`PARTIAL`: current approximation is `L=1` plus dynamics. A true flat
multi-horizon baseline with multiple horizon heads but no nested hierarchy is
not implemented yet.

```bash
.venv/bin/python examples/hierarchical_dreamer/make_ablation_config.py \
  --base "$BASE" \
  --output "$OUT/flat_single_level_dynamics.yaml" \
  --set hierarchical_latent.ablation_name=flat_single_level_dynamics \
  --set hierarchical_latent.num_heads=1 \
  --set hierarchical_latent.sparsity_topk='[48]' \
  --set hierarchical_latent.reconstruction.betas='[1.0]' \
  --set hierarchical_latent.sparse_dynamics.strides='[1]' \
  --set hierarchical_latent.sparse_dynamics.alphas='[1.0]'
```

### Dense Multi-Stride

`READY`: disables sparsity while keeping multiple levels and strides.

```bash
--set hierarchical_latent.ablation_name=dense_multi_stride \
--set hierarchical_latent.sparsity.mode=none \
--set hierarchical_latent.loss_weights.sparsity=0.0
```

### HarmonyDream External Baseline

Two paths exist:

- `READY`: XuanCe same-code HarmonyDream baseline, using the local DreamerV3
  learner with `harmony=True`.
- `PARTIAL`: official upstream JAX code, cloned locally and wrapped. It should
  run in its own JAX environment rather than being mixed into the XuanCe `.venv`.

Same-code XuanCe command:

```bash
ENV_ID=ALE/Breakout-v5 \
SEED=1 \
DEVICE=cuda:0 \
REPLAY_RATIO=0.25 \
RUN_NAME=XuanCe-HarmonyDream-Breakout-v5-seed1-100000steps-rr0.25 \
examples/hierarchical_dreamer/baselines/run_xuance_harmonydream_atari100k.sh
```

Official-code JAX command:

```bash
GAME=breakout \
SEED=0 \
DEVICE_ID=0 \
RUN_NAME=harmonydream-atari100k-breakout-seed0 \
examples/hierarchical_dreamer/baselines/run_harmonydream_atari100k.sh
```

Details and protocol caveats:
`examples/hierarchical_dreamer/baselines/HARMONYDREAM_BASELINE.md`.

### DyMoDreamer External Baseline

`PARTIAL`: official upstream code has been cloned locally and wrapped, but it
needs a separate dependency environment and a successful smoke run before it is
paper-ready.

```bash
GAME=breakout \
SEED=0 \
DEVICE=cuda:0 \
RUN_NAME=dymodreamer-atari100k-breakout-seed0 \
STEPS=4e5 \
TRAIN_RATIO=1024 \
EVAL_EVERY=1e4 \
EVAL_EPISODE_NUM=100 \
COMPILE=False \
examples/hierarchical_dreamer/baselines/run_dymodreamer_atari100k.sh
```

Details and protocol caveats:
`examples/hierarchical_dreamer/baselines/DYMODREAMER_BASELINE.md`.

### SGF External Baseline

`PARTIAL`: official upstream code has been cloned locally and wrapped. The
current `harmonydream` env only needs `wandb`, `torchvision`, and `ale_py` before
smoke testing.

```bash
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
AGENT_EVAL=all \
WM_EVAL=none \
AMP=True \
COMPILE=False \
PYTHON_BIN=/home/dat.tt2/miniconda3/envs/harmonydream/bin/python \
examples/hierarchical_dreamer/baselines/run_sgf_atari100k.sh
```

Details and protocol caveats:
`examples/hierarchical_dreamer/baselines/SGF_BASELINE.md`.

### EAWM External Baseline

`PARTIAL`: official upstream code has been uploaded locally and wrapped. Use a
dedicated `eadream` or `easimulus` environment before smoke testing.

Preferred Dreamer-style EADream run:

```bash
PYTHON_BIN=/home/dat.tt2/miniconda3/envs/eadream/bin/python \
GAME=breakout \
SEED=0 \
DEVICE=cuda:0 \
RUN_NAME=eadream-atari100k-breakout-seed0 \
examples/hierarchical_dreamer/baselines/run_eawm_eadream_atari100k.sh
```

Optional Simulus-based EAWM run with W&B:

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

Details and protocol caveats:
`examples/hierarchical_dreamer/baselines/EAWM_BASELINE.md`.

### Larger Flat Dreamer / T-SAE / Other External Methods

`PARTIAL`: T-SAE-style now has a same-code Dreamer-latent launcher; larger flat
Dreamer and several other external methods still need separate implementations
or adapters.

## External Baselines That Need User-Supplied Code or Specs

I will not silently reimplement these from paper names alone because small
implementation choices can change the comparison:

- HarmonyDream: XuanCe same-code baseline is runnable with local DreamerV3;
  official upstream code is also available locally and wrapped for Atari100K,
  but still needs a dedicated JAX environment and successful smoke run before
  marking paper-code-ready.
- DyMoDreamer: official upstream code is available locally and wrapped for
  Atari100K, but still needs a dedicated dependency environment and successful
  smoke run before marking paper-ready.
- SGF: official upstream code is available locally and wrapped for Atari100K,
  but still needs dependency completion and successful smoke run before marking
  paper-ready.
- EAWM: official upstream code is available locally and wrapped for Atari100K
  via both EADream and EASimulus. It still needs dedicated environments and
  smoke runs before marking paper-ready.
- T-SAE-style coarse temporal port: this row is in the paper because it is the
  nearest-method control for temporal sparse regularization without HTS-WM's
  action-conditioned multi-stride dynamics. Official upstream T-SAE code is
  available locally, but it trains sparse autoencoders on language-model
  activations rather than Dreamer latents. A same-code Dreamer-latent
  T-SAE-style adapter is available:

```bash
SEED=1 \
DEVICE=cuda:0 \
RUN_NAME=tsae-style-breakout-seed1 \
WANDB_MODE=online \
examples/hierarchical_dreamer/baselines/run_tsae_style_atari100k.sh
```

  See `examples/hierarchical_dreamer/baselines/TSAE_BASELINE.md`.
- EAWM, TPC, RePo, Denoised MDPs, DreamerPro, THICK, CW-VAE/MTS3, SPARTAN,
  EfficientZero V2, TD-MPC2: require official code, an adapter, or a precise
  reduced-scope baseline spec.

Runnable internal approximations are documented separately, but they should be
reported as approximations rather than as those external methods.

## Table Mapping

### tab:protocol - Evaluation Protocol

`PARTIAL`.

Runnable now for Atari100K-style single-game XuanCe runs:

```bash
ENV_ID=ALE/Breakout-v5 RUNNING_STEPS=100000 REPLAY_RATIO=0.25 \
CONFIG_FILE=config/generated_configs/full_htswm.yaml \
RUN_NAME=tab-protocol-atari-breakout-full \
examples/hierarchical_dreamer/train_ablation.sh
```

Missing for full table: Synthetic Multi-Timescale, Key-Door, Inventory,
Video-Background DMC, Crafter, DMLab wrappers/protocols, and aggregate script.

### tab:baselines - Core Baseline Taxonomy

`PARTIAL`.

Immediate runnable rows for Atari Breakout: Dreamer backbone, HTS-WM,
T-SAE-style port, XuanCe same-code HarmonyDream, DyMoDreamer official-code
wrapper, SGF official-code wrapper, and EAWM EADream official-code wrapper. Also
runnable through generated configs: flat single-level SAE approximation,
Matryoshka-only, recon-only hierarchy, dense multi-stride, no `L_temp`, and no
`L_vc`.

Missing rows: larger flat Dreamer, true flat multi-horizon, DyMoDreamer + HTS-WM.
EAWM EASimulus remains deferred until its dedicated env is solved.

### tab:baseline-execution-tiers - Baseline Execution Tiers

`PARTIAL`.

Use the same runnable subset as `tab:baselines`. P1/P2 comparators are not wired.
For omitted rows, record `not evaluated because implementation/assumption not
available in current XuanCe branch`.

### tab:main-results - Main Utility Summary

`PARTIAL`.

Can fill Atari/Dreamer/Matryoshka/HTS-WM cells from current runs. XuanCe
HarmonyDream, DyMoDreamer, SGF, and T-SAE-style port now have direct run
commands in the immediate queue above. EAWM is deferred until dependency setup is
stable. Memory, motion, distractor, DMC, and GPU-hour summaries need additional
suites and baselines.

### tab:hero-panel-slots - Hero Panel Slots

`MISSING` as a final figure/table because most suites are not implemented.
Atari100K slot can run now as a guardrail. Static-memory, multi-stage, and
dynamic-distractor slots need environments.

### tab:prefix - Prefix Refinement

`PARTIAL`.

Already logged:

- `hierarchical_latent/recon_loss_level_*`
- `hierarchical_latent/recon_marginal_gain_level_*`
- `hierarchical_latent/active_ratio`

Still missing as explicit table columns:

- prefix reward/probe utility
- per-prefix active feature count rather than global active ratio

Run source config:

```bash
--set hierarchical_latent.ablation_name=tab_prefix_full
```

### tab:level-horizon - Level-Horizon Prediction Error

`PARTIAL`.

Already logged at training strides:

- `hierarchical_latent/sdyn_loss_level_*`

Missing:

- evaluation error matrix at horizons `1,2,4,8,16,32` for every prefix, including
  horizons not used as training strides
- flat multi-horizon true baseline
- active-feature-normalized predictive efficiency

### tab:temporal-robustness - Boundary, Revisit, Nuisance

`MISSING`.

Current code can train temporal variants, but the table metrics are not logged:
Boundary F1, delay, false-change rate, revisitation similarity, nuisance
sensitivity.

Runnable training variants:

```bash
# no temporal
--set hierarchical_latent.temporal_consistency.mode=none

# smooth-only
--set hierarchical_latent.temporal_consistency.mode=smooth

# contrastive
--set hierarchical_latent.temporal_consistency.mode=contrastive
```

`far_negative_mode` is currently config tracking only; true no/hard/soft sampler
behavior still needs implementation.

### tab:ablation-plan - Core Ablation Plan

`PARTIAL`.

Covered axes:

- hierarchy depth
- stride schedule
- sparsity mode
- budget schedule via `sparsity_topk`
- temporal mode
- projector type
- VC mode
- gradient stop pathways
- training regime
- RSSM KL/free-bit knobs
- action conditioning

Missing/partial axes:

- adaptive sparse budget
- true far-negative modes
- separate trunks
- partial gradient scaling
- grouped loss harmonization
- action-summary encoder

### tab:collapse - Collapse Diagnostics

`PARTIAL`.

Runnable variants:

```bash
# no VC
--set hierarchical_latent.variance_covariance.mode=none \
--set hierarchical_latent.loss_weights.variance_covariance=0.0

# variance only
--set hierarchical_latent.variance_covariance.mode=variance

# covariance only
--set hierarchical_latent.variance_covariance.mode=covariance

# smooth only
--set hierarchical_latent.temporal_consistency.mode=smooth

# global TopK
--set hierarchical_latent.sparsity.mode=global_topk
```

Already logged: coarse projection std and VICReg components.

Also logged now for each level:

- `hierarchical_latent/level_*_active_ratio`
- `hierarchical_latent/level_*_alive_ratio`
- `hierarchical_latent/level_*_dead_ratio`
- `hierarchical_latent/level_*_min_variance`
- `hierarchical_latent/level_*_mean_variance`
- `hierarchical_latent/level_*_effective_rank`
- `hierarchical_latent/level_*_offdiag_cov_norm`
- `hierarchical_latent/level_*_utilization_entropy`

Still missing: full activation histograms as W&B histogram artifacts and
checkpoint-level feature tables.

### tab:compute - Compute Audit

`PARTIAL`.

Current logs now include:

- `compute/params_*`
- `compute/grad_norm_world_model`
- `compute/grad_norm_encoder`
- `compute/grad_norm_hierarchy`
- CUDA memory metrics when CUDA is available

Still missing: FLOPs/update, throughput, wall-clock train hours as a normalized
aggregate, inference latency, and diagnostic-only analysis cost.

### tab:cross-domain-protocol - Cross-Domain Protocol Audit

`MISSING` for final table. Atari single-game settings are runnable; full DMC,
Atari 26, memory, robustness, Crafter/Craftax, and DMLab protocols are not yet
registered in this HTS-WM launcher.

### tab:backbone-reproduction - Backbone Reproduction Audit

`PARTIAL`.

Dreamer baseline runs are available through `examples/dreamer_v3/dreamer_v3_atari.py`.
Need a reporting script that records reference score, reproduced score, config
hash, code commit, seed count, GPU, and environment wrapper differences.

### tab:claim-evidence-registry - Claim-to-Evidence Registry

`PARTIAL`.

Covered direct evidence partially: prefix error, stride-level losses, collapse
components.

Missing: downstream suites, event/boundary metrics, revisit similarity,
matched-param/FLOPs controls, nuisance robustness, second backbone, planner.

### tab:experiment-suite-matrix - Experiment-Suite Matrix

`MISSING` for most suites. Current code covers Atari-style XuanCe launcher only.

### tab:dreamer-backbone-audits - DreamerV3-Style Backbone Ablations

`PARTIAL`.

Runnable now:

```bash
# replay ratio sweep
for RR in 0.125 0.25 0.5 1.0; do
  REPLAY_RATIO=$RR RUN_NAME=backbone-replay-ratio-${RR}-breakout-seed1 ...
done

# model size knobs
--set world_model.recurrent_model.recurrent_state_size=256
--set world_model.recurrent_model.dense_units=256
--set world_model.discrete_size=16
--set world_model.stochastic_size=16

# KL/free-bit sweep
--set world_model.kl_dynamic=0.5
--set world_model.kl_representation=0.1
--set world_model.kl_free_nats=1.0
```

Missing: no-observation loss, no reward/value gradients, detached decoder,
imagination-horizon sweep if not exposed through config, normalization swaps.

### tab:dreamerv3-robustness-audit - DreamerV3 Robustness Technique Audit

`PARTIAL`.

Runnable now: KL balance, free bits, model-size-related latent knobs.

Missing or not exposed: symlog/no-symlog, return normalization disable, unimix
disable audit, AGC/optimizer audit, observation-gradient pathway variants,
reward/value-gradient pathway variants.

### tab:matched-controls - Parameter/Compute/Update-Matched Controls

`PARTIAL`.

Runnable controls: Dreamer anchor, dense multi-stride, static sparse hierarchy,
HTS-WM, flat single-level approximation.

Missing: parameter-matched larger flat, FLOPs-matched larger flat, true flat
partition, true flat multi-horizon, FLOPs/inference measurement scripts.

### tab:scaling-grid - Data, Replay, and Model Scaling

`PARTIAL`.

Runnable:

```bash
# environment data
for STEPS in 25000 50000 100000 200000; do
  RUNNING_STEPS=$STEPS ...
done

# replay ratio
for RR in 0.125 0.25 0.5 1.0; do
  REPLAY_RATIO=$RR ...
done

# model size
--set world_model.recurrent_model.recurrent_state_size=256
--set world_model.recurrent_model.dense_units=256

# hierarchy active budget
--set hierarchical_latent.sparsity_topk='[4,4,4,4,4,4]'
--set hierarchical_latent.sparsity_topk='[8,8,8,8,8,8]'
--set hierarchical_latent.sparsity_topk='[16,16,16,16,16,16]'

# sequence length
SEQ_LEN=32
SEQ_LEN=64
SEQ_LEN=128
```

Missing: automatic aggregation and imagination horizon exposure if not already
available in the config.

### tab:nearest-method-matrix - Nearest-Method Control Matrix

`PARTIAL`.

Same status as `tab:baselines`. HarmonyDream, DyMoDreamer, SGF, and EAWM
EADream now have official-code wrappers and run commands. T-SAE has official
language-side wrappers and a same-code Dreamer-latent port. Other external
prior-method baselines still need separate implementations or adapters.

### tab:offline-diagnosis - Offline Fixed-Buffer Diagnosis

`MISSING`.

Need fixed replay-buffer export, fixed-buffer loader, paired batch order, and
representation-only training/evaluation mode.

### tab:hyper-transfer - Hyperparameter Transfer Audit

`PARTIAL`.

Runnable sweeps:

```bash
# levels
--set hierarchical_latent.num_heads=1
--set hierarchical_latent.num_heads=2
--set hierarchical_latent.num_heads=4
--set hierarchical_latent.num_heads=6

# strides
--set hierarchical_latent.sparse_dynamics.strides='[32,16,8,4,2,1]'
--set hierarchical_latent.sparse_dynamics.strides='[6,5,4,3,2,1]'
--set hierarchical_latent.sparse_dynamics.strides='[1,1,1,1,1,1]' \
--set hierarchical_latent.sparse_dynamics.require_strict_decreasing=false

# loss weights
--set hierarchical_latent.loss_weights.hierarchical=0.1
--set hierarchical_latent.loss_weights.sparse_dynamics=0.1
--set hierarchical_latent.loss_weights.temporal=0.1
--set hierarchical_latent.loss_weights.variance_covariance=0.1

# backbone LR ratio
--set training_regime.encoder_lr_scale=0.0
--set training_regime.encoder_lr_scale=0.1
--set training_regime.encoder_lr_scale=1.0
```

Missing: adaptive budget and auxiliary linear warmup beyond staged two-phase.

### tab:rollout-fidelity - Open-Loop Fidelity

`MISSING`.

Training logs have world-model loss components, but no matched open-loop
evaluation script for latent/decoded error at horizons `1,4,8,16,32`.

### tab:dmc-task-results

`MISSING`.

Need DMC Visual launcher, selected task list, evaluation protocol, and table
aggregation.

### tab:atari-task-results

`PARTIAL`.

Single Atari games can run by changing `ENV_ID`; full 26-game command matrix and
HNS/IQM aggregation script are missing.

Example:

```bash
for GAME in Breakout Pong Boxing Frostbite Gopher RoadRunner; do
  ENV_ID=ALE/${GAME}-v5 RUN_NAME=atari-${GAME}-full-htswm-seed1 ...
done
```

### tab:planner-audit

`MISSING`.

No hierarchical planner is implemented. Keep this as future/P2 unless planning
efficiency becomes a claim.

## Figure Mapping

### fig:overview

`MISSING` as a paper figure, but no training run is required. Need diagram asset
showing Dreamer RSSM latent, sparse heads, nested reconstruction, multi-stride
dynamics, temporal/VC regularizers, and baseline contrast.

### fig:hero-results

`MISSING` for final multi-suite figure. Requires outputs from `tab:main-results`,
`fig:horizon-sweep`, and `fig:level-horizon`.

### fig:horizon-sweep

`MISSING`.

Needs controlled environment where dependency horizon/subgoal count can be
swept. Atari Breakout is not sufficient for this figure.

### fig:rliable-summary

`MISSING`.

Need aggregate script for mean, median, IQM, optimality gap, probability of
improvement, and performance profiles from multi-task results.

### fig:level-horizon

`PARTIAL`.

Training-stride losses are logged, but full heatmap requires an offline
evaluator over fixed horizons `1,2,4,8,16,32` and flat multi-horizon baseline.

### fig:prefix-refinement

`PARTIAL`.

Current level reconstruction losses and marginal reconstruction gains can support
a first version. Need optional reward/probe gain.

### fig:spliced-trajectory

`MISSING`.

Need trajectory selection/export, event labels, revisitation labels, nuisance
segments, and activation timeline plotting.

### fig:nuisance-event

`MISSING`.

Need paired clean/perturbed trajectories and nuisance/event sensitivity metrics.

### fig:collapse-dashboard

`PARTIAL`.

Use variants from `tab:collapse`. Need additional logging for effective rank,
min variance, covariance norm, alive/dead ratio, histograms, and TopK entropy.

### fig:compute-pareto

`MISSING`.

Need compute instrumentation from `tab:compute`.

### fig:loss-interactions

`PARTIAL`.

Already logged: raw and weighted hierarchy loss contributions. HarmonyDream's
official JAX code logs `harmony_s*`, `coeff*`, and `sigma*` in its own
`metrics.jsonl` once run through the external wrapper.

Also logged now: gradient norms for world-model, encoder, and hierarchy parameter
groups.

Missing: pairwise gradient cosine similarities for `L_WM`, `L_hier`, `L_sdyn`,
`L_temp`, `L_vc`, `L_sparse`; grouped harmonization.

### fig:prefix-rollouts

`MISSING`.

Need detached diagnostic decoder/export script for coarse, middle, and full
prefix rollouts.

### fig:factor-recovery

`MISSING`.

Needs controlled factor benchmark with macro-state/subgoal/pose/velocity/
distractor labels and probe scripts.

### fig:backbone-audit

`PARTIAL`.

Replay-ratio, model-size, and data-budget sweeps are runnable. Learning-signal
removal and normalization/optimizer audits are missing.

### fig:scaling-transfer

`PARTIAL`.

Can run level, sparse-budget, replay, data, and model-size sweeps. Needs
multi-suite transfer and aggregation scripts.

### fig:openloop-rollouts

`MISSING`.

Same blocker as `tab:rollout-fidelity`: no matched open-loop evaluator yet.

### fig:task-level-curves

`PARTIAL`.

W&B has train/test curves for individual runs. Need suite-level artifact
collector and plotting script with consistent axes and confidence intervals.

### fig:second-backbone

`MISSING`.

No second backbone port exists.

## Core Sweep Templates

### Training Regime

`READY`.

```bash
for REGIME in frozen_encoder two_phase fully_joint; do
  BASE_REGIME=examples/hierarchical_dreamer/config/atari100k_${REGIME}.yaml
  .venv/bin/python examples/hierarchical_dreamer/make_ablation_config.py \
    --base "$BASE_REGIME" \
    --output "$OUT/regime_${REGIME}.yaml" \
    --set hierarchical_latent.ablation_name=regime_${REGIME}

  CONFIG_FILE=config/generated_configs/regime_${REGIME}.yaml \
  RUN_NAME=regime-${REGIME}-breakout-seed1 \
  ENV_ID=$ENV_ID DEVICE=$DEVICE WANDB_MODE=$WANDB_MODE PROJECT_NAME=$PROJECT_NAME \
  RUNNING_STEPS=$RUNNING_STEPS REPLAY_RATIO=$REPLAY_RATIO \
  examples/hierarchical_dreamer/train_ablation.sh
done
```

### Levels

`READY` if list-valued fields are kept length-compatible.

```bash
# Example L=4.
.venv/bin/python examples/hierarchical_dreamer/make_ablation_config.py \
  --base "$BASE" \
  --output "$OUT/levels_4.yaml" \
  --set hierarchical_latent.ablation_name=levels_4 \
  --set hierarchical_latent.num_heads=4 \
  --set hierarchical_latent.sparsity_topk='[8,8,8,8]' \
  --set hierarchical_latent.reconstruction.betas='[1,1,1,1]' \
  --set hierarchical_latent.sparse_dynamics.strides='[8,4,2,1]' \
  --set hierarchical_latent.sparse_dynamics.alphas='[1,1,1,1]' \
  --set hierarchical_latent.sparsity.gammas='[0.0001,0.0001,0.0001,0.0001]'
```

### Stride Schedule

`READY`.

```bash
# geometric
--set hierarchical_latent.sparse_dynamics.strides='[32,16,8,4,2,1]'

# linear
--set hierarchical_latent.sparse_dynamics.strides='[6,5,4,3,2,1]'

# uniform negative control
--set hierarchical_latent.sparse_dynamics.strides='[1,1,1,1,1,1]' \
--set hierarchical_latent.sparse_dynamics.require_strict_decreasing=false
```

### Sparsity and Budget

`READY`.

```bash
# modes
--set hierarchical_latent.sparsity.mode=none
--set hierarchical_latent.sparsity.mode=l1
--set hierarchical_latent.sparsity.mode=level_topk
--set hierarchical_latent.sparsity.mode=global_topk

# budget schedules
--set hierarchical_latent.sparsity_topk='[8,8,8,8,8,8]'      # uniform
--set hierarchical_latent.sparsity_topk='[2,4,6,8,10,12]'    # coarse-small/fine-large
--set hierarchical_latent.sparsity_topk='[12,10,8,6,4,2]'    # coarse-large/fine-small
```

### Temporal, Projector, and VC

`READY` except true far-negative sampler.

```bash
# temporal
--set hierarchical_latent.temporal_consistency.mode=none
--set hierarchical_latent.temporal_consistency.mode=smooth
--set hierarchical_latent.temporal_consistency.mode=contrastive

# projector
--set hierarchical_latent.temporal_consistency.projector_type=none
--set hierarchical_latent.temporal_consistency.projector_type=linear
--set hierarchical_latent.temporal_consistency.projector_type=nonlinear

# VC
--set hierarchical_latent.variance_covariance.mode=none
--set hierarchical_latent.variance_covariance.mode=variance
--set hierarchical_latent.variance_covariance.mode=covariance
--set hierarchical_latent.variance_covariance.mode=both
```

### Gradient Flow

`PARTIAL`: stop-gradient switches are available; separate trunks and partial
gradient scaling are not.

```bash
# no stop-gradient
--set hierarchical_latent.reconstruction.stop_gradient_lower_levels=false \
--set hierarchical_latent.sparse_dynamics.stop_gradient_lower_levels=false \
--set hierarchical_latent.sparse_dynamics.target_stop_gradient=false

# reconstruction prefix SG only
--set hierarchical_latent.reconstruction.stop_gradient_lower_levels=true \
--set hierarchical_latent.sparse_dynamics.stop_gradient_lower_levels=false

# dynamics prefix SG only
--set hierarchical_latent.reconstruction.stop_gradient_lower_levels=false \
--set hierarchical_latent.sparse_dynamics.stop_gradient_lower_levels=true

# default
--set hierarchical_latent.reconstruction.stop_gradient_lower_levels=true \
--set hierarchical_latent.sparse_dynamics.stop_gradient_lower_levels=true \
--set hierarchical_latent.sparse_dynamics.target_stop_gradient=true
```

### Action Conditioning

`READY` except action-summary encoder.

```bash
--set hierarchical_latent.sparse_dynamics.action_mode=state_only
--set hierarchical_latent.sparse_dynamics.action_mode=current_action
--set hierarchical_latent.sparse_dynamics.action_mode=subsequence
```

## Missing Code/Metric Backlog

Highest priority to make the paper figures/tables producible:

1. Analysis evaluator for level-by-horizon error matrix and prefix reward/probe gain.
2. W&B histogram/artifact export for sparse activations and checkpoint-level
   feature tables.
3. True temporal negative sampler: none/hard/soft far negatives with terminal
   masks and revisitation-safe labels when available.
4. Boundary/revisitation/nuisance metric suite and spliced-trajectory export.
5. Compute instrumentation still missing: FLOPs/update, throughput, normalized
   wall-clock aggregation, inference latency.
6. Task suite launchers and aggregators: DMC Visual, Atari 26, controlled
   horizon/memory environments, video-background robustness.
7. Baseline adapters still missing: larger flat Dreamer, true flat multi-horizon,
   external P1 comparators. T-SAE-style temporal-only on Dreamer latents is
   implemented as a launcher but still needs validation runs.
8. Offline fixed-buffer protocol.
9. Open-loop rollout evaluator and detached diagnostic prefix decoder.
10. Plotting/aggregation scripts for RLiable-style metrics, task curves, and
    paper-ready figures.
