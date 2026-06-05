# 2026-06-05 Paper-Final Launcher And Setup Split

## Context

The updated paper requires separating development results from paper-final
results. Same-code Atari runs need final-checkpoint reporting, `100` eval
episodes, explicit seeds, and result artifacts that can be aggregated into paper
tables.

## Changes

- Added `examples/hierarchical_dreamer/PAPER_SETUP_TASKS.md`.
  - Splits remaining paper work into tasks that can be implemented immediately
    and tasks requiring user decisions, benchmark assets, or external setup.
- Added `examples/hierarchical_dreamer/run_paper_final_samecode_atari100k.sh`.
  - Defaults to `CHECKPOINT_RULE=final`.
  - Defaults to `TEST_EPISODE=100`.
  - Supports `METHODS=dreamer,htswm,harmony,tsae` or `METHODS=all`.
  - Supports comma-separated `SEEDS`, defaulting to `0,1,2,3,4`.
- Updated local same-code entrypoints:
  - `examples/dreamer_v3/dreamer_v3_atari.py`
  - `examples/hierarchical_dreamer/hierarchical_dreamer_atari.py`
- Added CLI support for:
  - `--seed`
  - `--test-episode`
  - `--checkpoint-rule best|final`
- Added eval JSON artifacts:
  - `<log_dir>/eval_results/best_eval.json`
  - `<log_dir>/eval_results/final_eval.json`
- Updated wrappers to pass seed, checkpoint rule, and eval episodes:
  - `examples/hierarchical_dreamer/train_ablation.sh`
  - `examples/hierarchical_dreamer/train_atari100k_all_regimes.sh`
  - `examples/hierarchical_dreamer/baselines/run_xuance_harmonydream_atari100k.sh`
  - `examples/hierarchical_dreamer/baselines/run_tsae_style_atari100k.sh`
- Added `examples/hierarchical_dreamer/analysis/collect_paper_results.py`.
  - Scans eval JSON and W&B summaries.
  - Writes `logs/paper_results/runs.csv`.
  - Writes `logs/paper_results/runs.md`.
- Updated `PAPER_EXPERIMENT_REGISTRY.md` and `PASTE_READY_BASELINE_RUNS.md`
  with the new paper-final workflow.

## Validation

Ran:

```bash
bash -n \
  examples/hierarchical_dreamer/run_paper_final_samecode_atari100k.sh \
  examples/hierarchical_dreamer/train_ablation.sh \
  examples/hierarchical_dreamer/train_atari100k_all_regimes.sh \
  examples/hierarchical_dreamer/baselines/run_xuance_harmonydream_atari100k.sh \
  examples/hierarchical_dreamer/baselines/run_tsae_style_atari100k.sh

.venv/bin/python -m py_compile \
  examples/dreamer_v3/dreamer_v3_atari.py \
  examples/hierarchical_dreamer/hierarchical_dreamer_atari.py \
  examples/hierarchical_dreamer/analysis/collect_paper_results.py

METHODS=none SEEDS=0 WANDB_MODE=disabled \
  examples/hierarchical_dreamer/run_paper_final_samecode_atari100k.sh

.venv/bin/python examples/hierarchical_dreamer/analysis/collect_paper_results.py \
  --root . \
  --output-dir logs/paper_results
```

## Notes

No long training run was launched. The paper-final launcher is ready for the
user to run with the desired methods, seeds, device, and W&B project.
