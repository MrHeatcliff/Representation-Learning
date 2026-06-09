# Paper Implementation TODO

This file tracks implementation work that must be finished before paper-final
runs. It intentionally excludes result collection and table-filling tasks.

## P0: HTS Method Correctness

- [ ] Replace current temporal hinge proxy with the paper-described temporal
  contrastive objective.
  - Implement local positive sampling.
  - Implement same-trajectory far negatives.
  - Implement cross-trajectory negatives.
  - Support `far_negative_mode = none | soft | hard`.
  - Apply terminal/reset masks so positives and negatives never cross episode
    boundaries.
- [ ] Add per-level objective weights.
  - `beta_l` for nested reconstruction.
  - `alpha_l` for sparse dynamics.
  - `gamma_l` for sparsity.
- [ ] Add official-native training regimes.
  - Frozen encoder.
  - Two-phase default.
  - Fully joint.
  - Ensure optimizer/LR handling is explicit and logged.
- [ ] Verify gradient routing.
  - Decoder-prefix stop-gradient.
  - Dynamics-prefix stop-gradient.
  - Optional separate trunk or gradient scaling if coarse levels are unstable.

## P0: Same-Code Ablations

- [ ] `flat_mh`: true flat multi-horizon baseline.
  - One flat projector.
  - Horizons `[1, 2, 4, 8, 16, 32]`.
  - Six predictor heads.
  - Same action-subsequence encoder as HTS.
- [ ] `sgf_style_flat_same_code`.
  - Flat projector.
  - One-step action-conditioned prediction.
  - Variance/covariance regularization.
  - No sparsity, hierarchy, multi-horizon, or contrastive loss.
- [ ] `recon_only_hierarchy`.
  - Dense/unconstrained six heads.
  - Prefix decoders and nested reconstruction.
  - Disable TopK, L1, sparse dynamics, temporal loss, VC, negatives.
- [ ] `dense_multistride_no_sparse`.
  - Keep hierarchy, prefixes, strides, action-subsequence predictors, temporal
    loss, and VC.
  - Disable only TopK/L1 sparsity.
- [ ] `flat_sae`.
  - Use `code_width = 192`.
  - Use `topk = 48`.
  - Do not report the old one-head approximation.
- [ ] `larger_flat_param`.
  - Search flat width from `flat_mh`.
  - Match HTS add-on parameter count within `+-2%`.
  - Save search artifact.
- [ ] `larger_flat_flops`.
  - Implement FLOPs estimator first.
  - Report both dense-equivalent and realized TopK estimates.

## P0: Diagnostics Needed By Paper Tables/Figures

- [ ] Prefix reconstruction evaluator.
  - Held-out normalized reconstruction error per prefix.
  - Marginal prefix gain.
  - Optional reward prediction error by prefix.
- [ ] Level-by-horizon evaluator.
  - Evaluate every prefix at horizons `{1, 2, 4, 8, 16, 32}`.
  - Report latent NRMSE and predictive utility per active feature.
- [ ] Collapse diagnostics.
  - Alive/dead features per level.
  - Per-dimension variance and minimum variance.
  - Off-diagonal covariance norm per level.
  - TopK utilization entropy.
  - Effective rank.
- [ ] Compute diagnostics.
  - FLOPs/update.
  - Peak accelerator memory.
  - Wall-clock train time.
  - Inference latency.
  - Throughput.
- [ ] Final checkpoint evaluator.
  - 100 evaluation episodes for Atari final tables.
  - Same checkpoint rule for all methods.
  - Raw per-episode scores and HNS.

## P1: Benchmark/Wrapper Work

- [ ] Synthetic Multi-Timescale fixed-buffer benchmark.
- [ ] Synthetic periodic evaluation and aggregation.
- [ ] MiniHack/NLE KeyCorridor wrapper.
- [ ] THICK-compatible milestone logging.
- [ ] DMC/DMC-GB2 wiring.
- [ ] Real plotting backend for curve grids and heatmaps.
- [ ] `rliable`/bootstrap aggregation for final multi-task summaries.

## Current Status

- Official DreamerV3 baseline logging has been extended with paper artifacts.
- Official HTS port exists and runs on Atari100K Alien.
- Current HTS temporal objective is still a hinge proxy, not final paper InfoNCE.
- Current Alien results are development evidence only, not paper-final evidence.
