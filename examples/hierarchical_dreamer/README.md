# Hierarchical Dreamer

This is the isolated research fork for modifying DreamerV3 without changing the baseline implementation.

Primary edit points:

- `xuance/torch/representations/hierarchical_dreamer.py`: world model and representation changes.
- `xuance/torch/policies/hierarchical_dreamer.py`: actor, critic, imagination, and policy logic.
- `xuance/torch/learners/model_based/hierarchical_dreamer_learner.py`: losses, optimizers, schedules, and training metrics.
- `xuance/torch/agents/model_based_rl/hierarchical_dreamer_agent.py`: component wiring and training-loop-level changes.
- `examples/hierarchical_dreamer/config/atari.yaml`: Atari100k-style experiment config.
- `examples/hierarchical_dreamer/hierarchical_dreamer_atari.py`: runnable Atari entrypoint.

Quick smoke test:

```bash
.venv/bin/python examples/hierarchical_dreamer/hierarchical_dreamer_atari.py \
  --env-id ALE/Pong-v5 \
  --device cpu \
  --running-steps 1 \
  --benchmark 0 \
  --wandb-mode disabled \
  --buffer-size 64 \
  --batch-size 2 \
  --seq-len 4
```

Online W&B run:

```bash
.venv/bin/python examples/hierarchical_dreamer/hierarchical_dreamer_atari.py \
  --config-file config/atari100k_two_phase.yaml
```

Atari100K regimes:

```bash
.venv/bin/python examples/hierarchical_dreamer/hierarchical_dreamer_atari.py \
  --config-file config/atari100k_frozen_encoder.yaml

.venv/bin/python examples/hierarchical_dreamer/hierarchical_dreamer_atari.py \
  --config-file config/atari100k_two_phase.yaml

.venv/bin/python examples/hierarchical_dreamer/hierarchical_dreamer_atari.py \
  --config-file config/atari100k_fully_joint.yaml
```

Sequential Atari100K run, including DreamerV3 baseline last:

```bash
examples/hierarchical_dreamer/train_atari100k_all_regimes.sh
```
