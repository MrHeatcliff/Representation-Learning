# 2026-05-28 - Temporal Consistency, VICReg, Sparsity, Full Objective

## Request

Add the remaining auxiliary objectives:

- Temporal contrastive consistency on the coarsest level `z_t^(1)`.
- VICReg-style variance and covariance regularization to prevent collapse.
- Per-level sparsity penalty.
- Full weighted objective:

```text
L = L_WM
  + lambda_hier L_hier
  + lambda_sdyn L_sdyn
  + lambda_temp L_temp
  + lambda_vc L_vc
  + lambda_sparse L_sparse
```

## Changes

- Added `TemporalContrastiveVICReg` in `xuance/torch/policies/hierarchical_dreamer.py`.
- Projection uses the coarsest latent head only:

```text
v_t = p_theta(z_t^(1))
```

- Temporal contrastive loss uses nearby states as positives:

```text
positive: v_{t + k}
```

- Negatives are the other candidate states in the InfoNCE batch, which includes other sampled trajectories and temporally distant states from the same trajectory.
- Added VICReg-style:

```text
variance term: keep projection dimensions active
covariance term: decorrelate dimensions
```

- Added L1 sparsity penalty per level:

```text
L_sparse = sum_l gamma_l ||z_t^(l)||_1
```

- Kept TopK sparsity level-wise in `SparseLatentHeads`; no global TopK is used.
- Added `loss_weights` in config so the full objective is explicit.

## Config

Added:

```yaml
hierarchical_latent:
  temporal_consistency:
    projection_hidden_size: 128
    projection_dim: 64
    positive_stride: 1
    temperature: 0.1
  variance_covariance:
    std_target: 1.0
    variance_weight: 1.0
    covariance_weight: 0.04
  sparsity:
    gammas: [0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001]
  loss_weights:
    hierarchical: 1.0
    sparse_dynamics: 1.0
    temporal: 0.1
    variance_covariance: 0.1
    sparsity: 1.0
```

## Logged Metrics

```text
model_loss/temporal_contrastive_loss
model_loss/variance_covariance_loss
model_loss/sparsity_loss
hierarchical_latent/temp_contrastive_loss
hierarchical_latent/vicreg_loss
hierarchical_latent/vicreg_variance_loss
hierarchical_latent/vicreg_covariance_loss
hierarchical_latent/coarse_projection_std
hierarchical_latent/sparsity_loss
hierarchical_latent/sparsity_loss_level_1
...
hierarchical_latent/sparsity_loss_level_L
```

## Verification

Compiled changed files:

```bash
.venv/bin/python -m py_compile \
  xuance/torch/policies/hierarchical_dreamer.py \
  xuance/torch/learners/model_based/hierarchical_dreamer_learner.py
```

Ran an Atari smoke test with backward and optimizer steps:

```bash
.venv/bin/python examples/hierarchical_dreamer/hierarchical_dreamer_atari.py \
  --env-id ALE/Pong-v5 \
  --device cpu \
  --running-steps 8 \
  --benchmark 0 \
  --wandb-mode disabled \
  --buffer-size 128 \
  --start-training 4 \
  --batch-size 2 \
  --seq-len 4
```

The smoke test completed with `Finish training!`.

Also ran a synthetic `TemporalContrastiveVICReg` module test; temporal and VICReg losses were finite and all expected metrics were emitted.
