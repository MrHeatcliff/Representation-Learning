# T-SAE Baseline Inspection

## Request

Clone and inspect `AI4LIFE-GROUP/temporal-saes`, preferably running it as an
official baseline if possible, and consider a separate environment to avoid
dependency conflicts.

## Changes

- Cloned the upstream T-SAE repository:
  - `external_baselines/temporal-saes`
  - commit `8daecb2ab39be81b7bc0f26ae62160f5fb38e822`
- Marked the external clone as ignored:
  - `external_baselines/temporal-saes/`
- Added baseline notes and separate-environment setup:
  - `examples/hierarchical_dreamer/baselines/TSAE_BASELINE.md`
- Added official-code helper scripts:
  - `examples/hierarchical_dreamer/baselines/run_tsae_language_smoke.sh`
  - `examples/hierarchical_dreamer/baselines/run_tsae_language_train.sh`
- Added a Dreamer-latent T-SAE-style baseline wrapper:
  - `examples/hierarchical_dreamer/baselines/run_tsae_style_atari100k.sh`
- Added `level_batch_topk` sparsity to `HierarchicalDreamerPolicy` to mirror
  the upstream T-SAE BatchTopK mechanism at each latent level.
- Updated the paper command map to distinguish official language-side T-SAE
  reproduction from the same-code Atari100K Dreamer-latent adapter.

## Findings

- Upstream T-SAE is a language-model activation SAE codebase using HuggingFace,
  `nnsight`, and Poetry.
- The official training entrypoint is
  `dictionary_learning/dictionary_learning/train_temporal.py`.
- It does not implement Atari environments, Dreamer world models, RL training,
  or episodic return evaluation.
- In the paper tables, T-SAE appears as a nearest-method control for temporal
  sparse regularization, not because the official repo is already an Atari
  agent baseline.
- The new same-code adapter launches that control by using reconstruction plus
  coarse temporal regularization and by setting the weighted sparse-dynamics loss
  to zero.
- CPU smoke run passed with `RUNNING_STEPS=1`, `BENCHMARK=0`,
  `WANDB_MODE=disabled`, confirming Atari env construction, generated config
  loading, policy/module initialization, and one training step.
- A separate `temporal-saes` conda environment is recommended because upstream
  uses Python `>=3.11 <3.13`, Poetry, and PyTorch `cu128` wheels.

## Remaining Work

- Install the dedicated T-SAE environment only if official language-side
  reproduction is needed.
- Run a GPU smoke/full Atari100K job for the Dreamer-latent `tsae_style` wrapper.
