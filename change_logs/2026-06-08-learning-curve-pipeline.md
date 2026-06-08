# 2026-06-08 Learning Curve Pipeline

## Changed

- Added unified paper artifact helpers for run metadata, resolved configs,
  periodic evaluation curves, final eval rows, config hashes, code commit, frame
  counts, optimizer updates, and Atari HNS values.
- Patched DreamerV3 and HierarchicalDreamer Atari entrypoints to write
  `artifacts/<experiment_id>/<suite>/<task>/<condition>/<method>/seed_<seed>/`.
- Added separate `--intermediate-test-episode` support so periodic curves can
  use fewer episodes than final checkpoint evaluation.
- Added paper pipeline scripts under `examples/hierarchical_dreamer/scripts/`.
- Added `PAPER_LEARNING_CURVE_PIPELINE.md` with audit findings, smoke commands,
  output mapping, and blockers.

## Validation

- Python compile passed for new scripts and modified Atari entrypoints.
- Synthetic smoke dataset, smoke checkpoint metrics, smoke tables, smoke PDFs,
  and smoke manifest were generated under `/tmp/htswm_paper_artifacts_smoke`.
- Artifact schema smoke generated `eval_curve.csv`, `final_eval.csv`,
  `learning_curve_auc.csv`, and `final_checkpoint_scores.csv`.

## Blockers

- The requested LaTeX manuscript file was not found in the workspace.
- MiniHack/NLE/rliable/matplotlib/pandas are not installed in `.venv`.
- Synthetic checkpoint evaluator and real curve plotting remain partial.
