# 2026-06-08 Atari100K 26-Game Queue

## Changes

- Added `examples/hierarchical_dreamer/paper_full_runs/ATARI100K_26GAME_BASELINE_QUEUE.md`.
- The queue contains paste-ready `bash <<'BASH'` blocks for the 26 Atari100K games with seeds `0..4`.
- Included ready methods: DreamerV3, HTS-WM, Flat-MH, Flat-SAE, SGF-style flat same-code, recon-only hierarchy, dense multi-stride no sparse, HTS no hierarchy, HTS no sparse dynamics, HTS no temporal consistency, HTS no variance/covariance, XuanCe HarmonyDream, and TSAE-style.
- Marked `larger_flat_param_breakout.yaml` as Breakout-only until per-game parameter-matching search is implemented.
- Updated `paper_full_runs/README.md` to point to the new queue file.
- Fixed stale generated-config path in `flat_mh_full_atari100k.md`.
- Added final-eval and video flags to existing DreamerV3 and Flat-MH full-run docs.
- Clarified how to disable final-eval video logging in the queue file.
