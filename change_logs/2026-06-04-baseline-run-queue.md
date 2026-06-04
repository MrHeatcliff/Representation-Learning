# Baseline Run Queue

## Request

Network setup for additional baselines is unstable, so update the paper command
guide with the baseline commands that can be started from the currently prepared
code and environments.

## Changes

- Added an immediate run queue to:
  - `examples/hierarchical_dreamer/ABLATION_COMMANDS.md`
- Included copy-paste commands for:
  - local XuanCe DreamerV3 anchor
  - full HTS-WM
  - T-SAE-style Dreamer-latent control
  - XuanCe same-code HarmonyDream
  - DyMoDreamer smoke and full Atari100K
  - SGF smoke and full Atari100K
- Marked EAWM as deferred until a dedicated `eadream` or `easimulus` env can be
  solved and smoke-tested.
- Updated table mapping text for `tab:baselines` and `tab:main-results` to
  reflect the current run queue.

## Follow-up

- Added EAWM EADream smoke and full Atari100K commands after the `eadream`
  environment and Atari ROM import were completed.
- Left EASimulus EAWM deferred until its separate environment is installed and
  smoke-tested.

## Notes

- SGF smoke uses smaller `WM_BATCH_SIZE` and `AGENT_BATCH_SIZE` to avoid the
  small-replay sampling error seen in short smoke runs.
- EAWM wrappers remain documented, but EAWM is not included in the immediate
  run queue because dependency setup was blocked by unstable network and solver
  conflicts.
