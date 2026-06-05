# Synthetic Multi-Timescale Benchmark

Fixed-buffer representation benchmark for HTS-WM mechanism experiments.

## Generate P0 State Dataset

```bash
cd /mnt/disk1/backup_user/dat.tt2/xuance

.venv/bin/python examples/hierarchical_dreamer/synthetic_multiscale/generate_dataset.py \
  --output data/synthetic_multiscale_state \
  --train-trajectories 10000 \
  --val-trajectories 2000 \
  --test-trajectories 2000 \
  --length 128 \
  --noise-std 0.01 \
  --seed 0
```

The dataset contains `.npz` shards plus `manifest.json`.

Stored arrays per shard:

| Key | Shape |
|---|---|
| `obs` | `(episodes, 128, 44)` for state-vector P0 |
| `actions` | `(episodes, 128)` discrete action ids |
| `rewards` | `(episodes, 128)` zero reward placeholder |
| `dones` | `(episodes, 128)` final step true |
| `factors` | `(episodes, 128, 5)` in order `fast, mid, slow, context, nuisance` |
| `boundaries` | `(episodes, 128, 5)` in order `fast, mid, slow, context, macro` |
| `macro_state_id` | `(episodes, 128)` |
| `full_state_id` | `(episodes, 128)` |
| `revisit_group_id` | `(episodes, 128)` currently equal to `macro_state_id` |
| `timestep` | `(episodes, 128)` |
| `episode_id` | `(episodes, 128)` |

## Generate P1 Pixel Dataset

```bash
cd /mnt/disk1/backup_user/dat.tt2/xuance

.venv/bin/python examples/hierarchical_dreamer/synthetic_multiscale/generate_dataset.py \
  --output data/synthetic_multiscale_pixel \
  --pixel \
  --train-trajectories 10000 \
  --val-trajectories 2000 \
  --test-trajectories 2000 \
  --length 128 \
  --seed 0
```

Pixel observations are deterministic renderings of the same latent factor
process with shape `(episodes, 128, 64, 64, 3)`.

## Locked Dynamics

Actions are five discrete ids mapped to signed increments:

```text
0 -> -2
1 -> -1
2 ->  0
3 -> +1
4 -> +2
```

Factors are categorical:

```text
fast      in Z_8
mid       in Z_8
slow      in Z_8
context   in Z_4
nuisance  in Z_16
```

Dynamics:

```text
fast[t+1]    = fast[t] + u_t mod 8
mid[t+1]     = mid[t] + sum(u[t-3:t+1]) mod 8 every 4 steps
slow[t+1]    = slow[t] + sign(sum(u[t-15:t+1])) mod 8 every 16 steps
context[t+1] = context[t] + 1 mod 4 every 64 steps
nuisance     = autonomous random walk with steps {-1,+1}
```

## Metrics To Implement Next

- level x horizon normalized prediction error;
- predictive quality per active feature;
- prefix reconstruction error and marginal prefix gain;
- factor probes;
- boundary F1 and delay;
- false-change rate;
- revisit similarity;
- effective rank;
- dead-feature ratio;
- nuisance sensitivity.
