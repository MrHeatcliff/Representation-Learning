# 2026-06-08 P0 Corrections

## Scope

Addressed the paper-lock P0 corrections before allowing paper-final runs.

## Changes

- Standardized same-code XuanCe `REPLAY_RATIO` defaults to `1` in ablation and XuanCe HarmonyDream launch scripts.
- Added true no/hard/soft far-negative handling for HTS temporal contrastive loss with episode-boundary masks from `is_first`.
- Fixed soft far-negative mode to downweight far same-trajectory negatives instead of treating the config as a tracking-only flag.
- Preserved `flat_sae` as one flat dictionary with width `192` and TopK budget `48`.
- Preserved true `flat_mh` as one flat latent with six action-conditioned predictors over horizons `[1, 2, 4, 8, 16, 32]`.
- Updated `dense_multistride_no_sparse` so it disables only TopK/L1/sparsity loss while keeping hierarchy, prefix reconstruction, multi-stride dynamics, temporal loss, and VC.
- Added generated configs for `hts_no_hier` and `hts_no_sdyn`.
- Updated markdown guidance to remove stale `REPLAY_RATIO=0.25` paper-final commands and stale far-negative TODO language.
- Added a paste-ready Breakout P0 smoke queue under `examples/hierarchical_dreamer/paper_smoke_runs/`.

## Validation

- `py_compile` passed for the hierarchical policy, Atari entrypoints, ablation config scripts, and paper artifact scripts.
- `bash -n` passed for the relevant training and artifact shell scripts.
- Tensor smoke test passed for temporal contrastive modes `none`, `hard`, and `soft` with episode-start masks.
- Tensor smoke test passed for flat controls: `flat_sae` produced width `192` with `48` active dimensions, `flat_mh` ran all six horizon heads, and `sgf_style_flat_same_code` produced predictive and VC losses.

## Still Not Launched

No full Atari, DMC, DMC-GB2, or external official-code campaigns were launched.
