# 2026-06-08 Paper Full-Run Commands

## Changed

- Added `examples/hierarchical_dreamer/paper_full_runs/`.
- Added paste-ready full Atari100K `26 games x 5 seeds` commands for:
  - DreamerV3 anchor;
  - Flat-MH same-code control.

## Notes

- These commands are paper-final oriented: final checkpoint, `100` eval episodes,
  `100000` action-level steps, seeds `0..4`.
- Additional method files should be added to this folder as each baseline becomes
  ready for full 26-game sweeps.
