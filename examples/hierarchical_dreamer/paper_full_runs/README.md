# Paper-Final Full-Run Commands

This folder contains paste-ready commands for paper-final experiment batches.
Use one file per method so each full 26-game sweep can be launched, monitored,
and tracked separately.

Paper-final Atari100K defaults:

```text
26 Atari games
seeds = 0,1,2,3,4
running_steps = 100000
checkpoint_rule = final
test_episode = 100
batch_size = 16
seq_len = 64
replay_ratio = 0.125
```

## Current Files

| File | Method | Status |
|---|---|---|
| `dreamerv3_full_atari100k.md` | DreamerV3 anchor | READY |
| `flat_mh_full_atari100k.md` | Flat multi-horizon same-code control | READY |

## Result Tracking

When a batch finishes, update:

- `examples/hierarchical_dreamer/RESULTS_TRACKER.md`
- `examples/hierarchical_dreamer/PAPER_EXPERIMENT_REGISTRY.md`

Then run the collector:

```bash
cd /mnt/disk1/backup_user/dat.tt2/xuance

.venv/bin/python examples/hierarchical_dreamer/analysis/collect_paper_results.py
```
