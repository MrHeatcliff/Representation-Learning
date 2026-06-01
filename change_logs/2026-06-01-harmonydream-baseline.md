# 2026-06-01 - HarmonyDream Baseline

## Request

Use the upstream HarmonyDream GitHub repository as a baseline if possible, or
adapt from it if running directly is not practical.

## Findings

- Cloned upstream repository locally:
  - `external_baselines/HarmonyDream`
  - commit `0a22e33fb47489361a9f239867a1cf34b958b320`
  - license MIT
- Upstream recommends `dreamerv3-jax` for Atari100K.
- The upstream Atari100K config uses an older DreamerV3 codebase and
  `run.train_ratio: 1024`.
- The local XuanCe code already contains a HarmonyDream-style harmonizer in
  DreamerV3 policy/learner, so we can run a same-code XuanCe Harmony baseline
  without porting the full upstream repository.

## Changes

- Added official-code wrapper:
  - `examples/hierarchical_dreamer/baselines/run_harmonydream_atari100k.sh`
- Added same-code XuanCe wrapper:
  - `examples/hierarchical_dreamer/baselines/run_xuance_harmonydream_atari100k.sh`
- Added notes and protocol caveats:
  - `examples/hierarchical_dreamer/baselines/HARMONYDREAM_BASELINE.md`
- Updated ablation command template to mark HarmonyDream as:
  - `READY` for XuanCe same-code baseline.
  - `PARTIAL` for official JAX baseline until dependencies are installed and a
    smoke run succeeds.
- Ignored the local upstream clone from the main git repo:
  - `external_baselines/HarmonyDream/`
- Fixed `examples/dreamer_v3/dreamer_v3_atari.py` so `harmony` no longer
  defaults to `True` via argparse and does not override YAML unless explicitly
  provided.

## Verification

- `bash -n` passed for both HarmonyDream wrapper scripts.
- `py_compile` passed for `examples/dreamer_v3/dreamer_v3_atari.py`.
- XuanCe HarmonyDream CPU smoke run completed with one update.

## Remaining

- Official JAX HarmonyDream still needs a separate JAX/Atari environment.
- Upstream JAX code logs JSON/TensorBoard by default; W&B is commented out in
  their `train.py`.
