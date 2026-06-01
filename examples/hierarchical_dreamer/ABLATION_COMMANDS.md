# HTS-WM Experiment Command Templates

This file maps the paper draft's tables, figures, and ablations to runnable
config flags. Commands use a two-step pattern:

1. Generate a YAML config with `make_ablation_config.py`.
2. Launch it with `train_ablation.sh`.

The launcher defaults to a current-official-DreamerV3-style update frequency:

```bash
REPLAY_RATIO=0.25
```

For paper-ratio alignment, use:

```bash
REPLAY_RATIO=0.125
```

## Common Setup

```bash
cd /mnt/disk1/backup_user/dat.tt2/xuance

BASE=examples/hierarchical_dreamer/config/atari100k_two_phase.yaml
OUT=examples/hierarchical_dreamer/config/generated_configs
DEVICE=cuda:0
WANDB_MODE=online
PROJECT_NAME=HTS-WM-Ablations
RUNNING_STEPS=100000
REPLAY_RATIO=0.25
```

## Full HTS-WM

```bash
.venv/bin/python examples/hierarchical_dreamer/make_ablation_config.py \
  --base "$BASE" \
  --output "$OUT/full.yaml" \
  --set hierarchical_latent.ablation_name=full

CONFIG_FILE=config/generated_configs/full.yaml \
RUN_NAME=full-two_phase-breakout-seed1 \
DEVICE=$DEVICE WANDB_MODE=$WANDB_MODE PROJECT_NAME=$PROJECT_NAME \
RUNNING_STEPS=$RUNNING_STEPS REPLAY_RATIO=$REPLAY_RATIO \
examples/hierarchical_dreamer/train_ablation.sh
```

## Table: Baseline Ladder

### Dreamer Backbone

Use the existing baseline entrypoint:

```bash
.venv/bin/python examples/dreamer_v3/dreamer_v3_atari.py \
  --env-id ALE/Breakout-v5 \
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

### Matryoshka-Only Port

Nested sparse reconstruction on, temporal/dynamics/VC off.

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

CONFIG_FILE=config/generated_configs/matryoshka_only.yaml \
RUN_NAME=matryoshka-only-breakout-seed1 \
DEVICE=$DEVICE WANDB_MODE=$WANDB_MODE PROJECT_NAME=$PROJECT_NAME \
RUNNING_STEPS=$RUNNING_STEPS REPLAY_RATIO=$REPLAY_RATIO \
examples/hierarchical_dreamer/train_ablation.sh
```

### Flat Multi-Horizon Control

This is approximated by `L=1` with sparse dynamics enabled. A true flat
multi-head implementation is still future work.

```bash
.venv/bin/python examples/hierarchical_dreamer/make_ablation_config.py \
  --base "$BASE" \
  --output "$OUT/flat_single_level_dynamics.yaml" \
  --set hierarchical_latent.ablation_name=flat_single_level_dynamics \
  --set hierarchical_latent.num_heads=1 \
  --set hierarchical_latent.sparsity_topk='[48]' \
  --set hierarchical_latent.reconstruction.betas='[1.0]' \
  --set hierarchical_latent.sparse_dynamics.strides='[1]' \
  --set hierarchical_latent.sparse_dynamics.alphas='[1.0]' \
  --set hierarchical_latent.sparsity.gammas='[0.0001]'

CONFIG_FILE=config/generated_configs/flat_single_level_dynamics.yaml \
RUN_NAME=flat-single-level-dynamics-breakout-seed1 \
DEVICE=$DEVICE WANDB_MODE=$WANDB_MODE PROJECT_NAME=$PROJECT_NAME \
RUNNING_STEPS=$RUNNING_STEPS REPLAY_RATIO=$REPLAY_RATIO \
examples/hierarchical_dreamer/train_ablation.sh
```

## Table: Training Regime

```bash
for REGIME in frozen_encoder two_phase fully_joint; do
  BASE_REGIME=examples/hierarchical_dreamer/config/atari100k_${REGIME}.yaml
  .venv/bin/python examples/hierarchical_dreamer/make_ablation_config.py \
    --base "$BASE_REGIME" \
    --output "$OUT/regime_${REGIME}.yaml" \
    --set hierarchical_latent.ablation_name=regime_${REGIME}

  CONFIG_FILE=config/generated_configs/regime_${REGIME}.yaml \
  RUN_NAME=regime-${REGIME}-breakout-seed1 \
  DEVICE=$DEVICE WANDB_MODE=$WANDB_MODE PROJECT_NAME=$PROJECT_NAME \
  RUNNING_STEPS=$RUNNING_STEPS REPLAY_RATIO=$REPLAY_RATIO \
  examples/hierarchical_dreamer/train_ablation.sh
done
```

## Table/Figure: Prefix Refinement

Sweep hierarchy depth.

```bash
for L in 1 2 3 4 6; do
  TOPK=$(python - <<PY
print([8] * int("$L"))
PY
)
  STRIDES=$(python - <<PY
L = int("$L")
base = [32, 16, 8, 4, 2, 1]
print(base[-L:])
PY
)
  .venv/bin/python examples/hierarchical_dreamer/make_ablation_config.py \
    --base "$BASE" \
    --output "$OUT/levels_${L}.yaml" \
    --set hierarchical_latent.ablation_name=levels_${L} \
    --set hierarchical_latent.num_heads="$L" \
    --set hierarchical_latent.sparsity_topk="$TOPK" \
    --set hierarchical_latent.reconstruction.betas="$(python - <<PY
print([1.0] * int("$L"))
PY
)" \
    --set hierarchical_latent.sparse_dynamics.strides="$STRIDES" \
    --set hierarchical_latent.sparse_dynamics.alphas="$(python - <<PY
print([1.0] * int("$L"))
PY
)" \
    --set hierarchical_latent.sparsity.gammas="$(python - <<PY
print([0.0001] * int("$L"))
PY
)"

  CONFIG_FILE=config/generated_configs/levels_${L}.yaml \
  RUN_NAME=levels-${L}-breakout-seed1 \
  DEVICE=$DEVICE WANDB_MODE=$WANDB_MODE PROJECT_NAME=$PROJECT_NAME \
  RUNNING_STEPS=$RUNNING_STEPS REPLAY_RATIO=$REPLAY_RATIO \
  examples/hierarchical_dreamer/train_ablation.sh
done
```

Note: replace placeholder lists with semantic values if sweeping non-uniform
betas/alphas/gammas. The generated command above is a template.

## Table/Figure: Level-Horizon Prediction

Stride schedule variants:

```bash
declare -A STRIDES
STRIDES[geometric]='[32,16,8,4,2,1]'
STRIDES[linear]='[6,5,4,3,2,1]'
STRIDES[uniform]='[1,1,1,1,1,1]'

for NAME in geometric linear uniform; do
  .venv/bin/python examples/hierarchical_dreamer/make_ablation_config.py \
    --base "$BASE" \
    --output "$OUT/stride_${NAME}.yaml" \
    --set hierarchical_latent.ablation_name=stride_${NAME} \
    --set hierarchical_latent.sparse_dynamics.strides="${STRIDES[$NAME]}" \
    --set hierarchical_latent.sparse_dynamics.require_strict_decreasing=false

  CONFIG_FILE=config/generated_configs/stride_${NAME}.yaml \
  RUN_NAME=stride-${NAME}-breakout-seed1 \
  DEVICE=$DEVICE WANDB_MODE=$WANDB_MODE PROJECT_NAME=$PROJECT_NAME \
  RUNNING_STEPS=$RUNNING_STEPS REPLAY_RATIO=$REPLAY_RATIO \
  examples/hierarchical_dreamer/train_ablation.sh
done
```

The template disables strict stride validation for this sweep so uniform stride
can be run as a negative control.

## Table: Sparsity and Budget

```bash
for MODE in none l1 level_topk global_topk; do
  .venv/bin/python examples/hierarchical_dreamer/make_ablation_config.py \
    --base "$BASE" \
    --output "$OUT/sparsity_${MODE}.yaml" \
    --set hierarchical_latent.ablation_name=sparsity_${MODE} \
    --set hierarchical_latent.sparsity.mode="$MODE"

  CONFIG_FILE=config/generated_configs/sparsity_${MODE}.yaml \
  RUN_NAME=sparsity-${MODE}-breakout-seed1 \
  DEVICE=$DEVICE WANDB_MODE=$WANDB_MODE PROJECT_NAME=$PROJECT_NAME \
  RUNNING_STEPS=$RUNNING_STEPS REPLAY_RATIO=$REPLAY_RATIO \
  examples/hierarchical_dreamer/train_ablation.sh
done
```

Budget schedules:

```bash
# uniform
--set hierarchical_latent.sparsity_topk='[8,8,8,8,8,8]'

# coarse-small / fine-large
--set hierarchical_latent.sparsity_topk='[2,4,6,8,10,12]'

# coarse-large / fine-small
--set hierarchical_latent.sparsity_topk='[12,10,8,6,4,2]'
```

## Table: Temporal and Far Negatives

```bash
for MODE in none smooth contrastive; do
  .venv/bin/python examples/hierarchical_dreamer/make_ablation_config.py \
    --base "$BASE" \
    --output "$OUT/temporal_${MODE}.yaml" \
    --set hierarchical_latent.ablation_name=temporal_${MODE} \
    --set hierarchical_latent.temporal_consistency.mode="$MODE"

  CONFIG_FILE=config/generated_configs/temporal_${MODE}.yaml \
  RUN_NAME=temporal-${MODE}-breakout-seed1 \
  DEVICE=$DEVICE WANDB_MODE=$WANDB_MODE PROJECT_NAME=$PROJECT_NAME \
  RUNNING_STEPS=$RUNNING_STEPS REPLAY_RATIO=$REPLAY_RATIO \
  examples/hierarchical_dreamer/train_ablation.sh
done
```

`far_negative_mode` is currently logged for config tracking. Implementing true
none/hard/soft far-negative sampling requires an additional sampler change.

## Table: Projector and VC

```bash
for PROJECTOR in none linear nonlinear; do
  EXTRA=""
  if [ "$PROJECTOR" = "none" ]; then
    EXTRA="--set hierarchical_latent.temporal_consistency.projection_dim=32"
  fi
  .venv/bin/python examples/hierarchical_dreamer/make_ablation_config.py \
    --base "$BASE" \
    --output "$OUT/projector_${PROJECTOR}.yaml" \
    --set hierarchical_latent.ablation_name=projector_${PROJECTOR} \
    --set hierarchical_latent.temporal_consistency.projector_type="$PROJECTOR" \
    $EXTRA
done

for VC in none variance covariance both; do
  .venv/bin/python examples/hierarchical_dreamer/make_ablation_config.py \
    --base "$BASE" \
    --output "$OUT/vc_${VC}.yaml" \
    --set hierarchical_latent.ablation_name=vc_${VC} \
    --set hierarchical_latent.variance_covariance.mode="$VC"
done
```

Launch each generated config with `train_ablation.sh`.

## Table: Gradient Flow

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

Separate trunks and partial gradient scaling are not implemented yet.

## Table: Action Conditioning

```bash
for ACTION_MODE in state_only current_action subsequence; do
  .venv/bin/python examples/hierarchical_dreamer/make_ablation_config.py \
    --base "$BASE" \
    --output "$OUT/action_${ACTION_MODE}.yaml" \
    --set hierarchical_latent.ablation_name=action_${ACTION_MODE} \
    --set hierarchical_latent.sparse_dynamics.action_mode="$ACTION_MODE"
done
```

Action-summary encoder is not implemented yet.

## Table: Objective Weights

```bash
# remove individual losses
--set hierarchical_latent.loss_weights.hierarchical=0.0
--set hierarchical_latent.loss_weights.sparse_dynamics=0.0
--set hierarchical_latent.loss_weights.temporal=0.0
--set hierarchical_latent.loss_weights.variance_covariance=0.0
--set hierarchical_latent.loss_weights.sparsity=0.0

# auxiliary warmup is represented by two_phase phase1_gradient_steps
--set training_regime.wm_only_phase1=true
--set training_regime.phase1_gradient_steps=20000
```

Grouped harmonization is not implemented for hierarchy losses yet.

## Table: RSSM Retuning

Use existing world-model keys:

```bash
--set world_model.kl_dynamic=0.5
--set world_model.kl_representation=0.1
--set world_model.kl_free_nats=1.0
--set world_model.recurrent_model.recurrent_state_size=512
--set world_model.recurrent_model.dense_units=512
--set world_model.discrete_size=32
--set world_model.stochastic_size=32
```

Changing RSSM size may require checkpoint isolation and fresh runs.

## Figures and Required Logged Signals

Already logged:

- `hierarchical_latent/recon_loss_level_*`
- `hierarchical_latent/sdyn_loss_level_*`
- `hierarchical_latent/active_ratio`
- `hierarchical_latent/mean_abs`
- `hierarchical_latent/coarse_projection_std`
- `hierarchical_latent/vicreg_*`
- `model_loss/*`
- `training_regime/*`
- train/test episode rewards

Still needed for full paper figures:

- level-by-horizon error matrix for horizons not equal to training strides
- prefix marginal reconstruction gain as an explicit metric
- activation histogram and TopK utilization entropy
- effective rank and per-dimension min variance
- boundary F1, delay, false-change rate
- revisitation similarity
- nuisance sensitivity
- GPU hours, peak VRAM, throughput, FLOPs estimate
- gradient norms and gradient cosine similarities per objective
