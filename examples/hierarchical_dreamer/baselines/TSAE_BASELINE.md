# T-SAE Baseline Notes

This baseline tracks the official Temporal SAE repository as an external
method reference. The actual paper row is a **T-SAE-style coarse temporal port**
inside the Dreamer-latent setting, not the upstream language-model experiment
unchanged.

## Source

- Repository: `https://github.com/AI4LIFE-GROUP/temporal-saes.git`
- Local clone: `external_baselines/temporal-saes`
- Inspected commit: `8daecb2ab39be81b7bc0f26ae62160f5fb38e822`
- License: Apache-2.0 for the top-level repo; the vendored
  `dictionary_learning` package declares MIT.

## Status

`PARTIAL` for official-code reproduction, `PARTIAL` for the paper's intended
Dreamer-latent control.

The upstream code trains Temporal Sparse Autoencoders on language-model
activations via HuggingFace, `nnsight`, and the Pile/FineWeb-style text
datasets. Its main training entrypoint is:

```text
dictionary_learning/dictionary_learning/train_temporal.py
```

It is not a Dreamer agent, does not construct Atari environments, and does not
produce episodic return metrics. Therefore it cannot directly fill the paper's
T-SAE row.

## Why T-SAE Appears In The Paper Baselines

The paper includes `T-SAE-style coarse temporal port` in `tab:baselines` and
`tab:nearest-method-matrix` as a nearest-method control for temporal sparse
regularization. Its role is to answer:

```text
If we add a T-SAE-like temporal objective to sparse latents, but do not add
HTS-WM's action-conditioned multi-stride dynamics, do we still get the same
timescale separation and return gains?
```

This is different from running the official T-SAE repository as-is. The upstream
repo establishes the reference method, but the fair comparison for this paper
requires a port that consumes Dreamer latent sequences `h_t`.

## Separate Environment

Use a dedicated environment so its Poetry, HuggingFace, `nnsight`, and torch
constraints do not conflict with HarmonyDream, DyMoDreamer, SGF, or XuanCe.

```bash
conda create -n temporal-saes python=3.11 -y
conda activate temporal-saes
python -m pip install --upgrade pip setuptools wheel poetry

cd /mnt/disk1/backup_user/dat.tt2/xuance/external_baselines/temporal-saes
poetry config virtualenvs.create false
poetry install
```

Upstream currently points torch packages to the PyTorch `cu128` wheel index. If
the CUDA driver cannot run CUDA 12.8 wheels, install a compatible torch build in
this environment and keep the exact package versions in the run log.

## Smoke Check

After installing the environment:

```bash
PYTHON_BIN=/home/dat.tt2/miniconda3/envs/temporal-saes/bin/python \
examples/hierarchical_dreamer/baselines/run_tsae_language_smoke.sh
```

This checks importability of torch, transformers, datasets, nnsight, and the
T-SAE trainer classes without downloading an LLM.

## Dreamer-Latent T-SAE-Style Baseline

Use this for the paper's `T-SAE-style coarse temporal port` row:

```bash
SEED=1 \
DEVICE=cuda:0 \
RUN_NAME=tsae-style-breakout-seed1 \
WANDB_MODE=online \
examples/hierarchical_dreamer/baselines/run_tsae_style_atari100k.sh
```

This adapter uses the local HierarchicalDreamer Atari training loop and applies:

- two sparse levels, corresponding to coarse/fine groups;
- ReLU latent activations;
- level-wise BatchTopK sparsity, matching the key upstream T-SAE sparsity
  mechanism more closely than per-sample TopK;
- prefix reconstruction;
- temporal contrastive regularization on the coarse level;
- optional VICReg-style variance/covariance regularization;
- no weighted multi-stride sparse-dynamics loss.

It is a same-code control for temporal sparse regularization, not an official
T-SAE result.

## Official Language Training Template

This runs upstream T-SAE on LLM activations. It is useful for method reference
or language-side reproduction, but not for Atari return tables.

```bash
PYTHON_BIN=/home/dat.tt2/miniconda3/envs/temporal-saes/bin/python \
MODEL=EleutherAI/pythia-160m \
LAYER=8 \
RUN_NAME=tsae-pythia160m-layer8-smoke \
MAX_STEPS=100 \
DISABLE_WANDB=true \
examples/hierarchical_dreamer/baselines/run_tsae_language_train.sh
```

Expected side effects:

- Downloads the HuggingFace model and streaming dataset if not cached.
- Writes checkpoints under `logs/external_baselines/tsae_language/<RUN_NAME>/`.
- Uses the upstream script's hardcoded `cuda:0` unless upstream code is patched.

## What Is Needed for Atari100K

To compare against HTS-WM, T-SAE must be adapted into this repo as a
Dreamer-latent baseline. A reasonable same-code baseline would:

1. Train or freeze a Dreamer world model and collect RSSM latent sequences
   `h_t`.
2. Train a temporal sparse autoencoder on those latents, not on LLM activations.
3. Use the same Atari100K data budget, seeds, replay buffer, evaluation
   protocol, and W&B/log folder conventions as the other regimes.
4. Use a T-SAE-style temporal loss on a coarse sparse split.
5. Deliberately omit HTS-WM's action-conditioned multi-stride dynamics, because
   that is the mechanism being tested.
6. Report reconstruction, sparsity, temporal smoothness, rank/collapse, and
   return metrics separately from HTS-WM's nested reconstruction and
   multi-stride dynamics.

This adapter is sufficient to launch the P0 control, but it should be reported
as `T-SAE-style port` rather than as the official language-model T-SAE method.
