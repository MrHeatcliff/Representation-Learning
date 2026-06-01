# 2026-06-01 - Paper Ablation Control Surface

## Summary

Added a flexible experiment control surface for HTS-WM ablations described in
`paper.txt`.

## Added

- `examples/hierarchical_dreamer/make_ablation_config.py`
  - Generates YAML configs from a base config and dotted `KEY=VALUE` overrides.
- `examples/hierarchical_dreamer/train_ablation.sh`
  - Launches one generated ablation config with standard Atari100K-style runtime
    arguments and W&B naming.
- `examples/hierarchical_dreamer/ABLATION_COMMANDS.md`
  - Command templates for the paper's baseline ladder, training regimes,
    hierarchy depth, stride schedules, sparsity modes, budget schedules,
    temporal modes, projector variants, VC modes, gradient-flow controls,
    action-conditioning controls, objective weights, and RSSM retuning.

## Config Flags

Added or formalized flags under `hierarchical_latent`:

- `ablation_name`
- `sparsity.mode`: `none`, `l1`, `level_topk`, `global_topk`
- `sparsity.budget_schedule`
- `sparsity_topk` as a per-level list
- `sparse_dynamics.require_strict_decreasing`
- `sparse_dynamics.target_stop_gradient`
- `sparse_dynamics.action_mode`: `state_only`, `current_action`, `subsequence`
- `temporal_consistency.mode`: `none`, `smooth`, `contrastive`
- `temporal_consistency.projector_type`: `none`, `linear`, `nonlinear`
- `temporal_consistency.far_negative_mode` for config tracking
- `variance_covariance.mode`: `none`, `variance`, `covariance`, `both`

## Logging

Added weighted hierarchy-loss contribution logs for objective-interaction plots:

- `model_loss/world_model_loss`
- `model_loss/weighted_hierarchical_recon_loss`
- `model_loss/weighted_sparse_dynamics_loss`
- `model_loss/weighted_temporal_loss`
- `model_loss/weighted_variance_covariance_loss`
- `model_loss/weighted_sparsity_loss`

## Known Gaps

The command template marks these as future implementation rather than runnable
claims:

- true flat multi-horizon baseline
- separate hierarchy trunks
- partial gradient scaling
- action-summary encoder
- true no/hard/soft far-negative sampler
- grouped harmonization for hierarchy losses
- boundary/revisitation/nuisance metrics
- gradient cosine similarity diagnostics
