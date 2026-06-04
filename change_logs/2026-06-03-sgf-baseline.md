# 2026-06-03 - SGF Baseline

## Request

Clone and analyze `jrobine/sgf.git`. Prefer running the upstream repository as a
baseline when possible; otherwise adapt from it.

## Findings

- Cloned upstream repository locally:
  - `external_baselines/sgf`
  - commit `11fb493b74f3b06ebaba8d71f302574967151568`
- License: MIT.
- README lists dependencies and a direct Atari training command.
- SGF is a compact official PyTorch Atari codebase with W&B logging.
- `configs/default.yaml` is already an Atari100K-style config with
  `trainer.env_steps: 100000`, RGB 64x64, `frame_skip: 4`, `sticky: false`, and
  final eval episodes `100`.
- The CLI does not expose budget fields directly; the wrapper generates a
  per-run config file from the upstream default config.

## Changes

- Added wrapper:
  - `examples/hierarchical_dreamer/baselines/run_sgf_atari100k.sh`
- Added baseline notes and setup instructions:
  - `examples/hierarchical_dreamer/baselines/SGF_BASELINE.md`
- Updated `examples/hierarchical_dreamer/ABLATION_COMMANDS.md` to mark SGF as an
  external-code `PARTIAL` baseline.
- Ignored the local upstream clone from the main git repo:
  - `external_baselines/sgf/`

## Verification

- `bash -n` passed for the SGF wrapper.
- Dependency check in the current `harmonydream` env:
  - present: `torch`, `gymnasium`, `ruamel.yaml`
  - missing: `torchvision`, `ale_py`, `wandb`
- SGF import currently fails only because `wandb` is missing.

## Remaining

- Install missing packages in `harmonydream`:
  - `pip install wandb torchvision ale-py moviepy`
- Run Atari import check.
- Run the smoke command from `SGF_BASELINE.md`.
- Run full Atari100K baseline after smoke succeeds.

## Follow-up

- In the reused `harmonydream` env, OpenCV/NumPy ABI compatibility matters.
  `opencv-python==4.8.1.78` requires NumPy 1.x, so the setup notes now pin
  `numpy==1.26.4` together with `opencv-python==4.8.1.78`.
- The smoke config needs smaller replay batch sizes than upstream defaults.
  The wrapper now exposes `WM_BATCH_SIZE` and `AGENT_BATCH_SIZE`.
