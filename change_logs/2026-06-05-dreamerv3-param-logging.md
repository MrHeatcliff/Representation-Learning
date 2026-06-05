# 2026-06-05 DreamerV3 Parameter Logging

## Context

The completed DreamerV3 baseline run did not include model parameter counts in
the W&B summary, making it harder to compare backbone size against HTS-WM and
external baselines.

## Changes

- Updated `xuance/torch/learners/model_based/dreamer_v3_learner.py`.
- Added static parameter counting to `DreamerV3_Learner`.
- Future DreamerV3 and HarmonyDream runs now log:
  - `compute/params_world_model`
  - `compute/params_world_model_trainable`
  - `compute/params_harmonizer`
  - `compute/params_harmonizer_trainable`
  - `compute/params_actor`
  - `compute/params_actor_trainable`
  - `compute/params_critic`
  - `compute/params_critic_trainable`
  - `compute/params_total`
  - `compute/params_total_trainable`

## Validation

Ran Python bytecode compilation for:

```bash
.venv/bin/python -m py_compile \
  xuance/torch/learners/model_based/dreamer_v3_learner.py \
  xuance/torch/learners/model_based/hierarchical_dreamer_learner.py
```

## Notes

This does not retroactively add parameter counts to existing W&B runs. It only
affects future runs launched after this code change.
