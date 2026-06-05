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

The launcher names W&B runs as:

```text
HDreamer-<regime>-<rom>-seed1-<steps>steps-<run_group>
DreamerV3-baseline-<rom>-seed1-<steps>steps-<run_group>
```

`DEVICE` defaults to the value in the launcher and can be overridden:

```bash
DEVICE=cuda:0 examples/hierarchical_dreamer/train_atari100k_all_regimes.sh
```

Paper ablation command templates:

```bash
less examples/hierarchical_dreamer/ABLATION_COMMANDS.md
```

Paper experiment registry and result tracker:

```bash
less examples/hierarchical_dreamer/PAPER_EXPERIMENT_REGISTRY.md
less examples/hierarchical_dreamer/RESULTS_TRACKER.md
```

The registry follows the updated `paper.txt` table and figure labels. The
tracker stores completed raw runs and marks whether each result is paper-final,
development-only, or official-code external.
