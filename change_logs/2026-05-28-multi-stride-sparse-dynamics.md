# 2026-05-28 - Multi-Stride Sparse Dynamics

## Request

Add the main hierarchical temporal objective:

```text
Delta_1 > Delta_2 > ... > Delta_L = 1
h_hat_{t + Delta_l} = F_l(z_t^(1), ..., z_t^(l), a_{t:t+Delta_l})
L_sdyn = sum_l alpha_l ||h_{t + Delta_l} - h_hat_{t + Delta_l}||_2^2
```

This is intended to avoid the failure mode where all levels learn only stride-1 details and coarse levels are ignored.

## Changes

- Added `MultiStrideSparseDynamics` in `xuance/torch/policies/hierarchical_dreamer.py`.
- Each level has its own predictor `F_l`.
- Each predictor receives:

```text
[z_t^(1), ..., z_t^(l), flattened action window]
```

- The action window uses the shifted Dreamer action convention:

```text
h_t -> h_{t+d} uses actions[t+1], ..., actions[t+d]
```

- Target future recurrent states are detached:

```text
target = stop_grad(h_{t + Delta_l})
```

- Coarser prefix heads are detached when training a finer level predictor:

```text
prefix_l = [stop_grad(z_t^(1)), ..., stop_grad(z_t^(l-1)), z_t^(l)]
```

- Added `L_sdyn` into the model loss in `HierarchicalDreamer_Learner.update()`.
- Added sparse dynamics parameters to `hierarchical_latent_parameters()`.
- Updated default hierarchy to 6 levels so strides can be strictly decreasing:

```yaml
num_heads: 6
sparse_dynamics:
  predictor_hidden_size: 512
  strides: [32, 16, 8, 4, 2, 1]
  alphas: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
  stop_gradient_lower_levels: true
```

## Logged Metrics

```text
model_loss/sparse_dynamics_loss
hierarchical_latent/sdyn_loss
hierarchical_latent/sdyn_loss_level_1
...
hierarchical_latent/sdyn_loss_level_L
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

Also ran a synthetic module test with sequence length 40 to exercise all configured strides `[32, 16, 8, 4, 2, 1]`; it produced a finite loss and all per-level metrics.
