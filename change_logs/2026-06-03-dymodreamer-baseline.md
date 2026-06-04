# 2026-06-03 - DyMoDreamer Baseline

## Request

Clone and analyze `Ultraman-Tiga1/DyMoDreamer.git`. Prefer running the upstream
repository as a baseline when possible; otherwise adapt from it.

## Findings

- Cloned upstream repository locally:
  - `external_baselines/DyMoDreamer`
  - commit `c40e4f428c5f7e6cd68acb2f576bf3a1072db4d9`
- No license file was found in the cloned repository.
- README is minimal and does not list dependencies.
- The repo has a PyTorch training entrypoint, `dreamer.py`, and an `atari100k`
  config in `configs.yaml`.
- The upstream train script references `dymodreamer/dreamer.py`, but the cloned
  repo stores `dreamer.py` at the root.
- The code imports environment modules as `envs.atari`, `envs.wrappers`, etc.,
  while those files are stored at repo root. The wrapper creates an `envs/`
  package view with symlinks at runtime.

## Changes

- Added wrapper:
  - `examples/hierarchical_dreamer/baselines/run_dymodreamer_atari100k.sh`
- Added baseline notes and setup instructions:
  - `examples/hierarchical_dreamer/baselines/DYMODREAMER_BASELINE.md`
- Updated `examples/hierarchical_dreamer/ABLATION_COMMANDS.md` to mark
  DyMoDreamer as an external-code `PARTIAL` baseline.
- Ignored the local upstream clone from the main git repo:
  - `external_baselines/DyMoDreamer/`

## Verification

- `bash -n` passed for the DyMoDreamer wrapper.
- Import check in the current XuanCe `.venv` failed because `ruamel` is missing,
  so no smoke run was attempted in that environment.

## Remaining

- Create a separate `dymodreamer` environment or install the required packages.
- Run Atari import test.
- Run the smoke command from `DYMODREAMER_BASELINE.md`.
- Run full Atari100K baseline after smoke succeeds.

## Follow-up

- DyMoDreamer upstream uses `ruamel.yaml.safe_load()`, which is removed in
  `ruamel.yaml>=0.18`. The setup notes now pin `ruamel.yaml<0.18`.
- DyMoDreamer boolean CLI parsing accepts only `False` and `True`, not lowercase
  `false` or `true`. The wrapper/docs now use capitalized boolean values.
