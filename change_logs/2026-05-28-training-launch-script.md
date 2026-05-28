# 2026-05-28 - Training Launch Script

## Summary

Added a bash launcher for the full Atari100K comparison run.

## Script

- `examples/hierarchical_dreamer/train_atari100k_all_regimes.sh`

## Run Order

1. Hierarchical Dreamer `frozen_encoder`
2. Hierarchical Dreamer `two_phase`
3. Hierarchical Dreamer `fully_joint`
4. DreamerV3 baseline

DreamerV3 is intentionally launched last.

## Logging

The script writes process logs under:

```text
logs/training_scripts/<run_group>/
```

Training outputs still use their own local W&B/checkpoint folders:

```text
logs/hierarchical-dreamer/atari100k/<regime>/<rom>/
models/hierarchical-dreamer/atari100k/<regime>/<rom>/
logs/dreamer-v3/atari100k/dreamer_baseline/<rom>/
models/dreamer-v3/atari100k/dreamer_baseline/<rom>/
```

## Runtime Overrides

The script supports environment-variable overrides such as:

```bash
DEVICE=cuda:0 WANDB_MODE=online RUNNING_STEPS=100000 \
  examples/hierarchical_dreamer/train_atari100k_all_regimes.sh
```
