# 2026-05-28 - Hierarchical Latent Heads

## Request

Add a Markdown logging area for major changes, then modify Hierarchical Dreamer so that each Dreamer recurrent latent state `h_t` passes through a trunk and multiple sparse latent heads:

```text
u_t = g_psi(h_t)
z_t^(ell) = sigma_ell(W_ell u_t + b_ell), ell = 1, ..., L
z_t = [z_t^(1), ..., z_t^(L)]
```

## Changes

- Added `change_logs/` for per-change Markdown notes.
- Added `SparseLatentHeads` in `xuance/torch/policies/hierarchical_dreamer.py`.
- Overrode `HierarchicalDreamerPolicy.model_forward()` so RSSM recurrent states `h_t` are transformed after `rssm.dynamic(...)` and before the Dreamer model state is built.
- Added a readout from concatenated sparse heads back to `h_t` dimension.
- Integrated the transformed recurrent state by residual update:

```text
h_t' = h_t + residual_scale * readout(z_t)
```

- Added W&B/log metrics:

```text
hierarchical_latent/active_ratio
hierarchical_latent/mean_abs
```

- Added `hierarchical_latent` config block to `examples/hierarchical_dreamer/config/atari.yaml`.
- Added hierarchical latent parameters to the Hierarchical Dreamer model optimizer.

## Notes

The implementation is isolated to `HierarchicalDreamer`; baseline `DreamerV3Policy` is not modified.

Design interpretation:

- `z_t^(1)` is the coarsest latent level.
- `z_t^(L)` is the finest latent level.
- Each level is represented by a separate sparse head.
- Separate heads give each level its own sparsity budget, preventing one level from consuming the full sparse capacity.
- Future hierarchy-specific objectives should preserve this coarse-to-fine ordering instead of treating heads as an unordered ensemble.

## Verification

Compiled the changed Python files:

```bash
.venv/bin/python -m py_compile \
  xuance/torch/policies/hierarchical_dreamer.py \
  xuance/torch/learners/model_based/hierarchical_dreamer_learner.py \
  examples/hierarchical_dreamer/hierarchical_dreamer_atari.py
```

Ran a short initialization smoke test:

```bash
.venv/bin/python examples/hierarchical_dreamer/hierarchical_dreamer_atari.py \
  --env-id ALE/Pong-v5 \
  --device cpu \
  --running-steps 1 \
  --benchmark 0 \
  --wandb-mode disabled \
  --buffer-size 64 \
  --start-training 1024 \
  --batch-size 2 \
  --seq-len 4
```

Ran a short update smoke test that exercised backward and optimizer steps:

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

Both smoke tests completed with `Finish training!`.
