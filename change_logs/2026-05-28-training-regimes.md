# 2026-05-28 - Atari100K Training Regimes

## Summary

Added three training regimes for Hierarchical Dreamer Atari100K experiments:

- `frozen_encoder`: encoder gradients are disabled and the encoder learning rate is zero.
- `two_phase`: default regime. Phase 1 trains only the standard Dreamer world-model objective; after the warmup, hierarchy objectives are enabled and encoder learning rate is smaller than hierarchy learning rate.
- `fully_joint`: world model encoder and hierarchy are optimized jointly with the same learning-rate scale.

## Implementation

- Added regime-specific optimizer parameter groups in `HierarchicalDreamer_Learner`:
  - `world_model`
  - `encoder`
  - `hierarchy`
- Added phase-1 gating so `two_phase` does not route recurrent states through the hierarchy during the world-model-only phase.
- Kept the standard world-model objective intact:
  - observation reconstruction loss
  - reward loss
  - continue/done loss
  - dynamic and representation KL losses
- Added W&B metrics for:
  - hierarchy loss components
  - sparse latent activity
  - training-regime phase state
  - encoder and hierarchy learning rates

## Configs

- `examples/hierarchical_dreamer/config/atari100k_frozen_encoder.yaml`
- `examples/hierarchical_dreamer/config/atari100k_two_phase.yaml`
- `examples/hierarchical_dreamer/config/atari100k_fully_joint.yaml`

Each config writes local logs under:

```text
logs/hierarchical-dreamer/atari100k/<regime>/Breakout-v5/
```

and model checkpoints under:

```text
models/hierarchical-dreamer/atari100k/<regime>/Breakout-v5/
```
