# 2026-06-01 - Paper Table/Figure Command Audit

## Request

Re-read the updated `paper.txt`, identify which paper tables and figures still
lack runnable support, and update the experiment command Markdown so each table
and figure is easy to check.

## Changes

- Rebuilt `examples/hierarchical_dreamer/ABLATION_COMMANDS.md` around the
  current paper labels.
- Added status markers for each item:
  - `READY`: runnable with current code.
  - `PARTIAL`: training can run but metrics, baselines, or plots are incomplete.
  - `MISSING`: needs new code, environments, baselines, or analysis scripts.
- Added a table-by-table mapping for all current `tab:*` labels in `paper.txt`.
- Added a figure-by-figure mapping for all current `fig:*` labels in
  `paper.txt`.
- Added explicit runnable command templates for the currently exposed control
  surface: training regimes, hierarchy depth, stride schedule, sparsity/budget,
  temporal/projector/VC, gradient flow, and action conditioning.
- Added a prioritized missing-code backlog for analysis metrics, controlled
  suites, compute instrumentation, external baselines, offline fixed-buffer
  diagnosis, rollout evaluation, and plot aggregation.

## Notes

- Atari single-game training is currently the most complete runnable path.
- Many paper rows now intentionally show as `PARTIAL` or `MISSING`; this is to
  prevent accidentally treating config-tracking flags or approximations as full
  paper-ready experiments.
- `far_negative_mode` remains a config-tracking field until the temporal negative
  sampler is implemented.
