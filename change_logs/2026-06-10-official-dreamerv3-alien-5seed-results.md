# 2026-06-10 Official DreamerV3 Alien 5-Seed Results

## Summary

Processed completed official DreamerV3 Atari100K Alien runs for seeds
`0,1,2,3,4` using the official clone with `--configs atari100k size12m`.

## Generated Artifacts

- `artifacts/paper_development/official_dreamerv3/alien_full26_size12m_5seeds/alien_dreamerv3_official_size12m_episode_curves.csv`
- `artifacts/paper_development/official_dreamerv3/alien_full26_size12m_5seeds/alien_dreamerv3_official_size12m_seed_summary.csv`
- `artifacts/paper_development/official_dreamerv3/alien_full26_size12m_5seeds/alien_dreamerv3_official_size12m_10bin_curve.csv`
- `artifacts/paper_development/official_dreamerv3/alien_full26_size12m_5seeds/alien_dreamerv3_official_size12m_5seed_10bin_curve.png`
- `artifacts/paper_development/official_dreamerv3/alien_full26_size12m_5seeds/alien_dreamerv3_official_size12m_5seed_10bin_curve.pdf`
- `artifacts/paper_development/official_dreamerv3/alien_full26_size12m_5seeds/alien_dreamerv3_official_size12m_5seed_10bin_hns_curve.png`
- `artifacts/paper_development/official_dreamerv3/alien_full26_size12m_5seeds/alien_dreamerv3_official_size12m_5seed_10bin_hns_curve.pdf`

## Aggregate

Using the last 10 online training episodes per seed:

- Mean score: `895.20`
- Std across seeds: `320.88`
- SEM: `143.50`
- Median score: `711.00`
- Mean HNS: `9.67`

## Paper Status

This result is valid for development learning curves and Atari curve sanity
checks. It is not a paper-final Atari task-table score because no separate final
`100`-episode evaluation artifact has been generated yet.

## Logging Fix

The completed runs were launched before the compact paper artifact fix, so their
`paper_artifacts/train_metrics.jsonl` files are large due to nested timer data.
The writer has been patched to exclude nested timer payloads for future runs.
