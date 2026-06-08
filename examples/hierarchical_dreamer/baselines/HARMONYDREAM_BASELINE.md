# HarmonyDream Baseline

This baseline uses the official upstream repository rather than a guessed
reimplementation inside XuanCe.

## Upstream

- Repository: `https://github.com/thuml/HarmonyDream`
- Local clone: `external_baselines/HarmonyDream`
- Inspected commit: `0a22e33fb47489361a9f239867a1cf34b958b320`
- License: MIT

The upstream repository contains:

- `dreamerv3-jax`: the path they explicitly recommend for Atari100K.
- `wmlib-torch`: PyTorch DreamerV2/HarmonyDream code for their visual suites.

For Atari100K, there are now two valid paths:

1. **Official-code baseline**: use upstream `dreamerv3-jax`. This is closest to
   their paper implementation.
2. **Same-code XuanCe baseline**: use XuanCe's existing HarmonyDream-style
   harmonizer inside the local DreamerV3 learner. This is more directly
   comparable to local Dreamer/HTS-WM runs, but it is not the official
   HarmonyDream codebase.

## What HarmonyDream Changes

In `dreamerv3-jax`, HarmonyDream adds three learnable scalar harmonizers:

- reward loss harmonizer
- image reconstruction loss harmonizer
- KL dynamics/representation harmonizer

The upstream code logs:

- `harmony_s1`, `harmony_s2`, `harmony_s3`
- `coeff1`, `coeff2`, `coeff3`
- `sigma1`, `sigma2`, `sigma3`

## XuanCe Same-Code Atari100K Command

This uses the local XuanCe DreamerV3 implementation with `harmony=True`.

```bash
cd /mnt/disk1/backup_user/dat.tt2/xuance

ENV_ID=ALE/Breakout-v5 \
SEED=1 \
DEVICE=cuda:0 \
REPLAY_RATIO=0.125 \
RUN_NAME=XuanCe-HarmonyDream-Breakout-v5-seed1-100000steps-rr1 \
examples/hierarchical_dreamer/baselines/run_xuance_harmonydream_atari100k.sh
```

This path logs to W&B through the local XuanCe logger and can be used in the
same project as Dreamer and HTS-WM runs.

For same-code fairness against current Dreamer/HTS-WM runs, use the canonical
XuanCe value `REPLAY_RATIO=0.125`. In this codebase, that means approximately one
minibatch update per agent step after `start_training`.

## Official-Code Atari100K Command

Wrapper:

```bash
cd /mnt/disk1/backup_user/dat.tt2/xuance

GAME=breakout \
SEED=0 \
DEVICE_ID=0 \
RUN_NAME=harmonydream-atari100k-breakout-seed0 \
PYTHON_BIN=python \
examples/hierarchical_dreamer/baselines/run_harmonydream_atari100k.sh
```

Equivalent upstream command:

```bash
cd external_baselines/HarmonyDream/dreamerv3-jax

CUDA_VISIBLE_DEVICES=0 python dreamerv3/train.py \
  --logdir /mnt/disk1/backup_user/dat.tt2/xuance/logs/external_baselines/harmonydream/harmonydream-atari100k-breakout-seed0 \
  --configs atari100k \
  --seed 0 \
  --task atari_breakout \
  --harmony True \
  --run.steps 1.01e5 \
  --run.train_ratio 1024 \
  --run.eval_every 5e3 \
  --run.eval_eps 100 \
  --jax.prealloc False \
  --jax.precision float32
```

## Important Protocol Differences

The official HarmonyDream Atari100K config is based on an older DreamerV3 commit
and uses:

- `run.steps: 1.01e5`
- `run.train_ratio: 1024`
- `batch_size: 16`
- `batch_length: 64`
- `env.atari.repeat: 4`
- `env.atari.sticky: False`
- `env.atari.gray: False`
- `env.atari.noops: 30`
- `env.atari.actions: needed`
- `rssm.deter: 512`
- CNN depth `32`
- MLP layers `2`
- units `512`

This is not identical to the current XuanCe Atari wrapper and not identical to
the later official DreamerV3 main-branch `train_ratio: 256` setting. Treat this
as an external-baseline run, not as a same-code ablation.

The local XuanCe Harmony path uses XuanCe's current Atari wrapper, replay-ratio
semantics, W&B logging, and model implementation. Use it for same-code fairness;
use the official JAX path for paper-code fidelity.

## Dependencies

Upstream README asks for old JAX/CUDA11-era dependencies:

```bash
cd external_baselines/HarmonyDream/dreamerv3-jax
pip install "jax[cuda11_cudnn82]"==0.4.6 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install -r requirements.txt
sh dreamerv3/embodied/scripts/install-atari.sh
```

On a modern CUDA 12/Python setup, it may be better to create a separate conda or
uv environment specifically for this baseline rather than installing these into
the XuanCe `.venv`.

## Logging

The upstream JAX code writes:

- `metrics.jsonl`
- `scores.jsonl`
- TensorBoard summaries
- `config.yaml`
- checkpoints

W&B is commented out in upstream `dreamerv3/train.py`. Use TensorBoard/JSON logs
first, then aggregate into the paper tables. Enabling W&B would require a small
upstream patch and should be kept separate from the faithful baseline run.

## Status

Runnable paths:

- XuanCe same-code HarmonyDream: `READY` once the local torch/CUDA environment is
  working.
- Official-code JAX HarmonyDream: `PARTIAL`; wrapper is ready, but execution
  depends on installing the upstream JAX Atari environment.

The official-code baseline should be marked paper-ready only after one smoke run
and one full Atari100K run complete successfully in its own environment.
