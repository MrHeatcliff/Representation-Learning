# 2026-06-09 RGB Atari Preprocessing

## Changes

- Added Atari preprocessing CLI overrides to DreamerV3 and HierarchicalDreamer entrypoints:
  - `--obs-type {ram,rgb,grayscale}`
  - `--img-size H W`
  - `--num-stack`
  - `--frame-skip`
- Threaded `OBS_TYPE`, `NUM_STACK`, `FRAME_SKIP`, `IMG_SIZE_0`, and `IMG_SIZE_1` through same-code Atari shell launchers.
- Added `examples/hierarchical_dreamer/paper_full_runs/DREAMERV3_SAMECODE_RGB_ATARI100K.md` with paste-ready Alien and 26-game RGB DreamerV3 commands.
- Updated same-code paper-final run names to include preprocessing tags such as `rgb-64x64-stack1-repeat4`.
- Added `--model-size {small,medium,large,xlarge}` to DreamerV3 and HierarchicalDreamer Atari entrypoints.
- Threaded `MODEL_SIZE` through same-code Atari launchers.
- Set the RGB DreamerV3 queue to `MODEL_SIZE=xlarge`, which is about 200.7M params in the XuanCe architecture.
- Updated paper-final log/model directories to include the preprocessing/model tag so RGB/xlarge runs do not overwrite grayscale/small runs.
- Added a dedicated RGB/xlarge 1K smoke queue for same-code baselines and ablations.
- Fixed paper-final and smoke config paths to use repo-root-relative `examples/hierarchical_dreamer/config/...` paths.
- Fixed the RGB/xlarge smoke queue repo-root detection from `paper_smoke_runs/`.
- Made `train_ablation.sh` normalize config paths to absolute paths when possible, so repo-root-relative and hierarchical-dreamer-relative config paths both work.
- Added `SMOKE_METHODS` filtering to the RGB/xlarge smoke queue so failed subsets can be rerun without repeating completed smoke runs.
- Ran RGB/xlarge 1K smoke checks for DreamerV3, HTS-WM full, Flat-SAE, and Flat-MH successfully. SGF-style flat reached 1000 training steps but was stopped before final eval to avoid sharing `cuda:0` with an active full Atari run.
- Updated the 26-game same-code baseline queue so every method block exports RGB Atari preprocessing and `MODEL_SIZE=xlarge`, and direct ablation run names include `rgb-xlarge`.
- Marked older method-specific full-run docs as legacy in the paper full-runs README; the canonical queue is now `ATARI100K_26GAME_BASELINE_QUEUE.md`.
- Changed same-code two-phase `phase1_gradient_steps` from `20000` to `2500` to match the updated `REPLAY_RATIO=0.125`; otherwise the hierarchy never activates within a 100K Atari100K run.

## RGB Sanity Check

- XuanCe Atari RGB env returns HWC observations with shape `(64, 64, 3)`.
- DreamerV3Agent transposes HWC to CHW before action/training.
- DreamerV3WorldModel receives CHW observation space `(3, 64, 64)`.
