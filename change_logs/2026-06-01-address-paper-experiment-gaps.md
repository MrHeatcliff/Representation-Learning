# 2026-06-01 - Address Paper Experiment Gaps

## Request

Separate gaps that depend on external baselines or paper-specific code from gaps
that can be addressed inside the current HTS-WM implementation, then implement
the addressable pieces.

## Implemented

- Added prefix marginal reconstruction gain logging:
  - `hierarchical_latent/recon_marginal_gain_level_*`
- Added sparse feature usage and collapse diagnostics per level:
  - active ratio
  - alive/dead ratio
  - mean absolute activation
  - min/mean variance
  - effective rank
  - off-diagonal covariance norm
  - utilization entropy
- Added compute/fairness diagnostics:
  - parameter counts for world model, hierarchy, actor, critic, and total
  - trainable parameter counts
  - model/encoder/hierarchy gradient norms
  - CUDA allocated/reserved memory when CUDA is available
- Updated `examples/hierarchical_dreamer/ABLATION_COMMANDS.md` to mark these
  items as partially addressed and to explicitly list external baselines that
  need user-supplied code, official implementations, or precise specs.

## Still Missing

- External baseline implementations/adapters such as Harmony Dreamer,
  DyMoDreamer, SGF-style flat, T-SAE-style port, EAWM, TPC, RePo, Denoised MDPs,
  DreamerPro, THICK, CW-VAE/MTS3, SPARTAN, EfficientZero V2, and TD-MPC2.
- Environment/protocol work for controlled horizon, memory, revisitation,
  nuisance, DMC, Crafter/Craftax, DMLab, and Atari 26 aggregation.
- Offline evaluators for full level-by-horizon heatmaps, boundary metrics,
  revisitation similarity, nuisance sensitivity, open-loop rollouts, and
  RLiable-style aggregation.
- FLOPs/update, throughput, normalized wall-clock aggregation, and inference
  latency instrumentation.

## Verification

- `py_compile` passed for the modified policy and learner.
- A CPU smoke run with one update completed successfully.
