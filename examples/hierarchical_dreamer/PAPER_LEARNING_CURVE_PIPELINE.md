# Detailed Learning Curve Pipeline

This document tracks the infrastructure requested by the latest detailed
learning-curve manuscript.

## Stage 0 Audit

| Item | Finding |
|---|---|
| Framework | PyTorch/XuanCe DreamerV3 and HierarchicalDreamer same-code entrypoints |
| Current runner structure | Example scripts own train/eval loops; DreamerV3 agent exposes `train()` and `test()` |
| Config mechanism | YAML loaded by `load_yaml`, CLI overrides via `recursive_dict_update`, then `argparse.Namespace` |
| Logging backend | W&B or TensorBoard via XuanCe agent logger; W&B is used for current paper runs |
| Checkpoint format | `Agent.save_model(...)` writes learner checkpoints under configured model dir |
| Evaluation hooks | Atari scripts support `--eval-protocol {periodic,final,train_only}`; paper/default launcher uses uninterrupted training plus final eval |
| Available env deps in `.venv` | `gymnasium`, `numpy`, `yaml` |
| Missing deps in `.venv` | `minihack`, `nle`, `rliable`, `matplotlib`, `pandas` |
| Manuscript file | `paper.txt`; verified learning-curve labels are present |
| Existing runs | W&B summaries/final JSON exist for some Breakout dev runs, but old runs lack unified `eval_curve.csv` |
| Main conflict fixed | Periodic eval is no longer required for paper runs; final eval/video is separate from training-episode return logging |

## Artifact Schema

New Atari final-eval runs write:

```text
artifacts/<experiment_id>/<suite>/<task>/<condition>/<method>/seed_<seed>/
  run_meta.json
  resolved_config.yaml
  eval_curve.csv
  final_eval.csv
```

Every `eval_curve.csv` row includes:

```text
experiment_id, suite, task, condition, method, seed, checkpoint_path,
global_step, env_steps, agent_actions, frames, action_repeat,
optimizer_updates, eval_episodes, metric_name, metric_value,
raw_or_normalized, config_hash, code_commit, wall_clock_seconds
```

For paper-aligned Atari learning curves, use training episode return logs from
W&B (`train/episode_return`) or local `train_episode_returns.csv` over
environment frames. Separate final eval artifacts are used for final score
tables and videos. For development runs that explicitly set
`EVAL_PROTOCOL=periodic`, `eval_curve.csv` can also be aggregated. For Atari,
both raw return and HNS are stored when the task has a normalization reference.

Dreamer-style curve aggregation:

```text
1. For each seed, bin completed training episodes by `env_frame`.
2. Compute mean episode return inside each seed/bin.
3. Compute mean and std across seeds per bin.
4. Use 30 bins over 0..400K env frames for Atari100K-style plots.
```

Aggregate local training episode CSVs:

```bash
cd /mnt/disk1/backup_user/dat.tt2/xuance
.venv/bin/python examples/hierarchical_dreamer/scripts/aggregate_train_episode_curves.py \
  --input-root logs \
  --output artifacts/curves/train_episode_return_curves.csv \
  --bins 30 \
  --max-env-frames 400000
```

## Implemented Scripts

| Script | Status |
|---|---|
| `scripts/generate_synthetic_multiscale.py` | READY |
| `scripts/run_periodic_eval.py` | PARTIAL: Atari integrated; Synthetic points to evaluator |
| `scripts/evaluate_synthetic_checkpoints.py` | SMOKE ONLY: structural placeholder metrics; model checkpoint integration pending |
| `scripts/extract_keycorridor_milestones.py` | SCHEMA/DEPS CHECK ONLY: blocked on MiniHack/NLE/THICK wrapper |
| `scripts/aggregate_learning_curves.py` | READY for separate-eval CSV seed aggregation |
| `scripts/aggregate_train_episode_curves.py` | READY for Dreamer-style training episode return curves |
| `scripts/build_paper_figures.py` | SMOKE ONLY: minimal PDFs without plotting deps |
| `scripts/build_paper_tables.py` | READY for final score/AUC/inventory CSVs |
| `scripts/build_paper_manifest.py` | READY, but direct LaTeX label scan needs manuscript file |
| `search_larger_flat_param.py` | READY |

## Smoke Commands

Generate a tiny synthetic dataset and smoke metrics:

```bash
cd /mnt/disk1/backup_user/dat.tt2/xuance

rm -rf /tmp/htswm_synth_pipeline_smoke artifacts/smoke_synthetic

.venv/bin/python examples/hierarchical_dreamer/scripts/generate_synthetic_multiscale.py \
  --output /tmp/htswm_synth_pipeline_smoke \
  --train-trajectories 4 \
  --val-trajectories 2 \
  --test-trajectories 2 \
  --length 16 \
  --shard-size 2 \
  --seed 11

.venv/bin/python examples/hierarchical_dreamer/scripts/evaluate_synthetic_checkpoints.py \
  --dataset-root /tmp/htswm_synth_pipeline_smoke \
  --output-root artifacts/smoke_synthetic/synthetic_multiscale/default/hts_full/seed_0 \
  --num-eval-checkpoints 3 \
  --smoke
```

Build smoke paper tables, figures, and manifest:

```bash
cd /mnt/disk1/backup_user/dat.tt2/xuance

.venv/bin/python examples/hierarchical_dreamer/scripts/build_paper_tables.py \
  --all \
  --artifact-root artifacts/smoke_synthetic \
  --output-root /tmp/htswm_paper_artifacts_smoke

.venv/bin/python examples/hierarchical_dreamer/scripts/build_paper_figures.py \
  --all \
  --artifact-root artifacts/smoke_synthetic \
  --output-root /tmp/htswm_paper_artifacts_smoke

.venv/bin/python examples/hierarchical_dreamer/scripts/build_paper_manifest.py \
  --artifact-root artifacts/smoke_synthetic \
  --paper-tex paper.txt \
  --output /tmp/htswm_paper_artifacts_smoke/manifest.json
```

## LaTeX Label Outputs

| Label | Output |
|---|---|
| `fig:keycorridor-learning` | `paper_artifacts/figures/keycorridor_learning_curves.pdf` |
| `fig:keycorridor-milestones` | `paper_artifacts/figures/keycorridor_milestone_curves.pdf` |
| `fig:synthetic-training` | `paper_artifacts/figures/synthetic_training_diagnostics.pdf` |
| `fig:hts-ablation-learning` | `paper_artifacts/figures/hts_component_ablation_curves.pdf` |
| `fig:atari100k-curves` | `paper_artifacts/figures/atari100k_task_level_curves.pdf` |
| `fig:dmc-visual-curves` | `paper_artifacts/figures/dmc_visual_task_level_curves.pdf` |
| `fig:dmcgb2-learning` | `paper_artifacts/figures/dmcgb2_robustness_curves.pdf` |

## Remaining Blockers

- Keep `paper.txt` updated; the manifest uses it as the manuscript source.
- Install/setup MiniHack/NLE and a THICK-compatible wrapper for KeyCorridor.
- Replace synthetic smoke metrics with real checkpoint-indexed representation
  evaluator.
- Add plotting backend or custom plotting implementation for real curve grids.
- Add `rliable` or an equivalent bootstrap implementation for final reliable
  aggregation.
- DMC, DMC-GB2, and condition-aware evaluation are not wired yet.
