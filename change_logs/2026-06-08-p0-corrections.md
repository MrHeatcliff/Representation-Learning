# 2026-06-08 P0 Corrections

## Scope

Addressed the paper-lock P0 corrections before allowing paper-final runs.

## Changes

- Standardized same-code XuanCe `REPLAY_RATIO` defaults to `0.125`, matching the DreamerV3 paper replay budget under XuanCe's minibatch-update semantics.
- Added true no/hard/soft far-negative handling for HTS temporal contrastive loss with episode-boundary masks from `is_first`.
- Fixed soft far-negative mode to downweight far same-trajectory negatives instead of treating the config as a tracking-only flag.
- Preserved `flat_sae` as one flat dictionary with width `192` and TopK budget `48`.
- Preserved true `flat_mh` as one flat latent with six action-conditioned predictors over horizons `[1, 2, 4, 8, 16, 32]`.
- Updated `dense_multistride_no_sparse` so it disables only TopK/L1/sparsity loss while keeping hierarchy, prefix reconstruction, multi-stride dynamics, temporal loss, and VC.
- Added generated configs for `hts_no_hier` and `hts_no_sdyn`.
- Updated markdown guidance to remove stale `REPLAY_RATIO=0.25` paper-final commands and stale far-negative TODO language.
- Added a paste-ready Breakout P0 smoke queue under `examples/hierarchical_dreamer/paper_smoke_runs/`.
- Fixed direct-script imports for DreamerV3 and Hierarchical Atari entrypoints by adding the repository root to `sys.path` before importing shared paper artifact helpers.
- Added optional W&B eval video upload for Atari entrypoints; final eval videos are enabled by default in smoke and paper-final launchers and appear as `Videos_Test`.
- Added W&B aliases `eval/episode_return_mean`, `eval/episode_return_std`, `eval/episode_return_min`, and `eval/episode_return_max` for return curves.
- Added `--eval-protocol {periodic,final,train_only}` to Atari entrypoints. Smoke and paper-final launchers now default to `final`, which trains uninterrupted and runs only one final separate eval/video.
- Added W&B aliases `train/episode_return` and `train/episode_length` when training episodes finish, matching the DreamerV3 Atari100K paper-style learning curve source.
- Refined training episode logging for Dreamer-style curves: added `train/agent_step`, `train/env_frame`, `train/action_repeat`, changed step reward to `train/episode_reward_mean`, and wrote local `train_episode_returns.csv`.
- Added `scripts/aggregate_train_episode_curves.py` to bin episode returns per seed over environment frames before computing mean/std across seeds.
- Verified the P0 Breakout smoke queue completed for 14 conditions, produced final eval JSONs, local training episode CSVs, W&B `Videos_Test` media, and aggregate training episode curves.

## Validation

- `py_compile` passed for the hierarchical policy, Atari entrypoints, ablation config scripts, and paper artifact scripts.
- `bash -n` passed for the relevant training and artifact shell scripts.
- Tensor smoke test passed for temporal contrastive modes `none`, `hard`, and `soft` with episode-start masks.
- Tensor smoke test passed for flat controls: `flat_sae` produced width `192` with `48` active dimensions, `flat_mh` ran all six horizon heads, and `sgf_style_flat_same_code` produced predictive and VC losses.

## Still Not Launched

No full Atari, DMC, DMC-GB2, or external official-code campaigns were launched.
