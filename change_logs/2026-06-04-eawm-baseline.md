# EAWM Baseline Inspection

## Request

Use the official `MarquisDarwin/EAWM` repository as an external baseline if
possible; otherwise adapt from it. Consider a separate environment to avoid
conflicts.

## Changes

- Accepted the user-uploaded EAWM clone at:
  - `external_baselines/EAWM`
- Inspected commit:
  - `d37f1ffa6c9dd42c163ed9e29e717bbafa399f37`
- Marked the external clone ignored:
  - `external_baselines/EAWM/`
- Added baseline notes and setup instructions:
  - `examples/hierarchical_dreamer/baselines/EAWM_BASELINE.md`
- Added official-code wrappers:
  - `examples/hierarchical_dreamer/baselines/run_eawm_eadream_atari100k.sh`
  - `examples/hierarchical_dreamer/baselines/run_eawm_easimulus_atari100k.sh`

## Findings

- EAWM provides two official implementations:
  - `EADream`: DreamerV3-style, best first target for this repo's EAWM baseline.
  - `EASimulus`: Simulus-based, native W&B logging, useful as a secondary
    official-code baseline.
- EAWM is GPL-3.0 licensed.
- `EADream` requires an old Python 3.9 / gym 0.19 / atari-py stack.
- `EASimulus` requires a separate Python 3.10 / Gymnasium Atari / Hydra stack.
- The uploaded clone shows many modified files inside its own git status,
  likely from Windows line-ending conversion. This is contained in the ignored
  external baseline directory.

## Remaining Work

- Create `eadream` and/or `easimulus` conda environments.
- Run import smoke checks.
- Run one short Atari smoke job before launching full Atari100K baselines.

## Follow-up

- Updated the EADream setup notes to use an Atari-only environment path and to
  avoid installing DMC/MemoryMaze dependencies that trigger a MuJoCo source
  build.
- Added the pinned `dm-control==1.0.9`, `dm-env==1.6`, and `mujoco==2.3.5`
  imports needed because EADream imports `dm_control` at module load time even
  for Atari runs.
- Patched local `external_baselines/EAWM/EADream/envs/wrappers.py` so
  `dm_control` and `mjbindings` are imported lazily only when DMC/DMC-GB wrappers
  need them. This avoids EGL/OSMesa import failures for Atari-only EADream runs.
- Imported the bundled EAWM Atari ROMs into the `eadream` environment using
  `atari_py.import_roms`; `breakout` is now present in `atari_py.list_games()`.
