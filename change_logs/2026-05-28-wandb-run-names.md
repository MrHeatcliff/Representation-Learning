# 2026-05-28 - W&B Run Names

## Summary

Changed W&B run names from timestamp-only names to descriptive experiment names.

## Changes

- `xuance/torch/agents/base/agent.py` now uses `config.wandb_run_name` when present.
- `examples/hierarchical_dreamer/hierarchical_dreamer_atari.py` auto-generates names with:

```text
<agent>-<training_regime>-<env>-seed<seed>-<steps>steps
```

- `examples/dreamer_v3/dreamer_v3_atari.py` auto-generates baseline names with:

```text
<agent>-baseline-<env>-seed<seed>-<steps>steps
```

- `examples/hierarchical_dreamer/train_atari100k_all_regimes.sh` passes explicit names including the launcher `RUN_GROUP`:

```text
HDreamer-<regime>-<rom>-seed1-<steps>steps-<run_group>
DreamerV3-baseline-<rom>-seed1-<steps>steps-<run_group>
```

## Device Note

Previous smoke tests used `--device cpu` intentionally to avoid relying on the local CUDA driver during short validation runs. Full training through the launcher uses the launcher `DEVICE` setting and can be overridden with:

```bash
DEVICE=cuda:0 examples/hierarchical_dreamer/train_atari100k_all_regimes.sh
```
