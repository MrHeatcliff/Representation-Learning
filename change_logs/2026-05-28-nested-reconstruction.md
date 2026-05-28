# 2026-05-28 - Nested Reconstruction

## Request

Add nested reconstruction for Hierarchical Dreamer:

```text
h_hat_t^(ell) = D_ell(z_t^(1), ..., z_t^(ell))
L_hier = sum_ell beta_ell ||h_t - h_hat_t^(ell)||_2^2
```

The intended semantics are:

- `z_t^(1)` is the coarsest level and should reconstruct broad information about `h_t`.
- Adding `z_t^(2), z_t^(3), ...` should refine missing details.
- When training decoder level `ell`, gradients from lower/coarser levels are stopped so fine levels do not pull coarse representations apart.

## Changes

- Added `NestedReconstructionDecoders` in `xuance/torch/policies/hierarchical_dreamer.py`.
- Each decoder `D_ell` receives the flattened prefix `[z_t^(1), ..., z_t^(ell)]`.
- Added per-level stop-gradient:

```text
prefix_ell = [stop_grad(z_t^(1)), ..., stop_grad(z_t^(ell-1)), z_t^(ell)]
```

- The target recurrent state `h_t` is detached in reconstruction loss so this auxiliary objective trains the sparse hierarchy and decoders to represent `h_t`, instead of moving the target itself.
- Added `L_hier` into the model loss in `HierarchicalDreamer_Learner.update()`.
- Added nested reconstruction parameters to the model optimizer through `hierarchical_latent_parameters()`.
- Added config:

```yaml
hierarchical_latent:
  reconstruction:
    decoder_hidden_size: 512
    betas: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    stop_gradient_lower_levels: true
```

## Logged Metrics

```text
model_loss/hierarchical_recon_loss
hierarchical_latent/recon_loss
hierarchical_latent/recon_loss_level_1
...
hierarchical_latent/recon_loss_level_L
```

## Verification

Compiled changed files:

```bash
.venv/bin/python -m py_compile \
  xuance/torch/policies/hierarchical_dreamer.py \
  xuance/torch/learners/model_based/hierarchical_dreamer_learner.py
```

Ran an update smoke test with backward and optimizer steps:

```bash
.venv/bin/python examples/hierarchical_dreamer/hierarchical_dreamer_atari.py \
  --env-id ALE/Pong-v5 \
  --device cpu \
  --running-steps 4 \
  --benchmark 0 \
  --wandb-mode disabled \
  --buffer-size 64 \
  --start-training 1 \
  --batch-size 2 \
  --seq-len 2
```

The smoke test completed with `Finish training!`.
