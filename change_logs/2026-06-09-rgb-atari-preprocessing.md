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
- Fixed a phase-switch crash in HTS temporal contrastive loss by normalizing `episode_starts` masks to `[T, B]`.
  The remote `error.txt` run crashed at about `21031/100000`, which is exactly when the new two-phase schedule activates the hierarchy.
  The fix accepts `[T, B]`, `[B, T]`, and `[T, B, 1]` masks before building same-episode and far-negative temporal pairs.
- Verified the fix with a direct temporal VICReg shape test and a real phase-switch smoke run using `phase1_gradient_steps=1`.
- Kept the paper-final same-code model size at `xlarge` as requested, but corrected the remaining Atari protocol defaults:
  `REPLAY_RATIO=1.0`, RGB observations, `repeat_action_probability=0.0`, `clip_reward=false`, and `episodic_life=false`.
- Added Atari env flags for sticky action probability, reward clipping, and episodic-life handling to DreamerV3 and HierarchicalDreamer Atari entrypoints and same-code launchers.
- Updated Atari configs and generated same-code ablation configs to use RGB, sticky-off, raw reward, no episodic-life boundaries, and `replay_ratio: 1.0`.
- Restored two-phase `phase1_gradient_steps` to `20000` so the phase switch remains near 21K agent steps under `REPLAY_RATIO=1.0`.
- Added debug/protocol W&B metrics for `agent_step`, `env_frame`, `gradient_step`, effective replay ratio, sticky probability, reward clipping, episodic-life mode, life-loss count, game-over count, reset count, truncation count, raw reward mean, and training reward mean.
- Confirmed the reward and critic heads already use symlog/two-hot distributions, so raw reward training is consistent with the DreamerV3-style objective.
- Switched the immediate Alien curve target from the 200M `xlarge` run to the HarmonyDream Atari100K reproduction target:
  `MODEL_SIZE=small`, `REPLAY_RATIO=1.0`, periodic eval every `5000` policy steps, and `100` eval episodes.
- Added `examples/hierarchical_dreamer/paper_full_runs/run_harmonydream_alien_curve.sh` and
  `examples/hierarchical_dreamer/paper_full_runs/HARMONYDREAM_ALIEN_CURVE.md` for the Alien-first DreamerV3 Reproduced vs HarmonyDream check.
- Guarded the HarmonyDream Alien curve launcher against inherited `MODEL_SIZE=xlarge`; it now uses `HARMONY_MODEL_SIZE=small` by default.
- Added fair core parameter-count logging for Dreamer-style runs: world model, actor, critic, target critic, and `params/core_agent_total`.
  The current XuanCe small Alien agent reports `19,116,691` core parameters.
- After comparing against the official `danijar/dreamerv3` Alien curve, corrected the XuanCe Atari100K parity path:
  manual action repeat in the wrapper with Gym ALE `frameskip=1`, 2-frame max pooling, Pillow resize, no reset autostart by default,
  `size12m` model preset matching official `deter=2048`, `hidden/units=256`, `stoch=32`, `classes=16`,
  and Alien curve default `REPLAY_RATIO=0.25` to match official `train_ratio=256` with `16 x 64` replay timesteps per update.

## RGB Sanity Check

- XuanCe Atari RGB env returns HWC observations with shape `(64, 64, 3)`.
- DreamerV3Agent transposes HWC to CHW before action/training.
- DreamerV3WorldModel receives CHW observation space `(3, 64, 64)`.
