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
| `ATARI100K_26GAME_BASELINE_QUEUE.md` | Canonical RGB/small 26-game queue for all currently ready same-code baselines | READY |
| `DREAMERV3_SAMECODE_RGB_ATARI100K.md` | XuanCe DreamerV3 with RGB Atari preprocessing | READY |
| `dreamerv3_full_atari100k.md` | Legacy DreamerV3 anchor; prefer the RGB/small files above | LEGACY |
| `flat_mh_full_atari100k.md` | Legacy Flat-MH control; prefer the RGB/small queue above | LEGACY |

Before launching RGB/small sweeps, run:

```bash
cat examples/hierarchical_dreamer/paper_smoke_runs/RGB_XLARGE_1K_SMOKE_COMMANDS.md
```

## Result Tracking

When a batch finishes, update:

- `examples/hierarchical_dreamer/RESULTS_TRACKER.md`
- `examples/hierarchical_dreamer/PAPER_EXPERIMENT_REGISTRY.md`

Then run the collector:

```bash
cd /mnt/disk1/backup_user/dat.tt2/xuance

.venv/bin/python examples/hierarchical_dreamer/analysis/collect_paper_results.py
```
