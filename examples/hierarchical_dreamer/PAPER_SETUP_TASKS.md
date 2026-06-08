# Paper Setup Tasks

This file splits the remaining paper work into tasks that can be implemented
from the current repo context and tasks that require user-provided choices,
external setup, or benchmark assets.

For the suite-by-suite baseline scope, see
`examples/hierarchical_dreamer/PAPER_BASELINE_MATRIX.md`.

## Group A: Can Be Implemented From Current Repo Context

| Task | Status | Output |
|---|---|---|
| Add paper-final same-code launcher mode: final checkpoint, `100` eval episodes | DONE | `examples/hierarchical_dreamer/run_paper_final_samecode_atari100k.sh` |
| Expose `TEST_EPISODE` and `CHECKPOINT_RULE` in local same-code wrappers | DONE | `train_ablation.sh`, `train_atari100k_all_regimes.sh`, Harmony/T-SAE wrappers |
| Expose `SEED` through local same-code entrypoints and wrappers | DONE | DreamerV3, HTS-WM, Harmony, T-SAE-style, regime runner |
| Write final/best eval JSON artifacts for local DreamerV3 and HTS-WM entrypoints | DONE | `<log_dir>/eval_results/{final,best}_eval.json` |
| Add W&B param logging for future DreamerV3/Harmony runs | DONE | `xuance/torch/learners/model_based/dreamer_v3_learner.py` |
| Create paper experiment registry by LaTeX table/figure label | DONE | `PAPER_EXPERIMENT_REGISTRY.md` |
| Track raw run completion separately from paper-final readiness | DONE | `RESULTS_TRACKER.md` |
| Add paste-ready paper-final same-code Atari commands | DONE | `PASTE_READY_BASELINE_RUNS.md` |
| Add basic HNS/rliable post-processing scripts | TODO | future `analysis/` scripts |
| Add basic result collector from eval JSON/W&B summaries | DONE | `analysis/collect_paper_results.py`, outputs `logs/paper_results/runs.{csv,md}` |
| Add task-level table generator from collected result rows | TODO | future `analysis/` scripts |
| Add compute extractor for params/runtime/memory/FLOPs/latency | TODO | future `analysis/` scripts |
| Add HTS-WM diagnostic extractors for prefix, horizon, collapse metrics | TODO | future `analysis/` scripts |
| Add Synthetic Multi-Timescale fixed-buffer dataset generator | DONE | `examples/hierarchical_dreamer/synthetic_multiscale/generate_dataset.py` |
| Add Synthetic fixed-buffer representation evaluator | TODO | prefix, horizon, probes, boundary, revisit, collapse metrics |
| Generate ablation configs for P0 controls already expressible by flags | TODO | generated configs + command blocks |
| Implement true same-code P0 controls not expressible by previous HTS flags | DONE/PARTIAL | `flat_mh`, `sgf_style_flat_same_code`, `flat_sae`, `larger_flat_param` done; `larger_flat_flops` deferred P1 |

## Group B: Requires User Input, External Setup, Or Benchmark Assets

| Task | Needed From User | Blocks |
|---|---|---|
| Choose paper seed set | RESOLVED: main benchmarks use `0,1,2,3,4`; dev sweeps use `0,1,2`; MiniHack final uses `0..6` | paper-final same-code runs |
| Decide whether SGF appears as a separate task-table column | RESOLVED: use same-code `SGF-style flat`; official SGF goes to appendix/external-reference table | `tab:atari-task-results`, appendix tables |
| Freeze hero figure suite slots | RESOLVED: DMC Visual full, Atari 100K full, DMC-GB2, MiniHack memory; panel b KeyCorridor-N; panel c Synthetic heatmap | `fig:hero-results`, `tab:hero-panel-slots` |
| Choose first controlled memory benchmark | RESOLVED: Synthetic first, then MiniHack KeyCorridor-N, then VisualPinPad, then Multiworld-Door | `fig:horizon-sweep`, `tab:temporal-robustness` |
| Provide/install controlled benchmark environments | MiniHack/DMC/DMC-GB2 repos/assets/install constraints | memory and robustness tasks |
| Provide DMC / DMC-GB2 setup decision | exact tasks/shifts and env installation | robustness/DMC tables |
| Decide scope for P1 external baselines | RESOLVED: required THICK, EAWM, RePo, DreamerPro, TD-MPC2; optional TPC, MTS3, Denoised MDPs; skip SPARTAN/full EfficientZero/full second-backbone/planner unless claims expand | P1 baseline tables |
| Provide external baseline repos/envs where network/setup is unstable | local archives or setup approval | official-code baseline runs |
| Decide whether planner claims remain in scope | RESOLVED: no planner claim for current scope; planner audit stays P2 / not applicable | `tab:planner-audit`, P2 work |

## Final Scope Decisions

| Question | Decision |
|---|---|
| Main paper-final seeds | `0,1,2,3,4` |
| Development/screening seeds | `0,1,2` |
| MiniHack final seeds | `0,1,2,3,4,5,6` |
| Official SGF in main Atari table | No. Use same-code `SGF-style flat`; keep official SGF in appendix/external reference. |
| Hero panel a | DMC Visual full, Atari 100K full, DMC-GB2, MiniHack memory. |
| Hero panel b | MiniHack KeyCorridor-N sweep, `N=4..11`. |
| Hero panel c | Synthetic Multi-Timescale level x horizon heatmap. |
| Controlled benchmark order | Synthetic -> MiniHack KeyCorridor-N -> VisualPinPad -> Multiworld-Door. |
| Required P1 external baselines | THICK, EAWM, RePo, DreamerPro, TD-MPC2. |
| Optional P1 external baselines | TPC, MTS3, Denoised MDPs. |
| Skipped for current scope | SPARTAN, full EfficientZero V2 reproduction, full second-backbone campaign, hierarchical planner. |

## Practical Run Order

| Phase | Suite | Methods | Outputs |
|---|---|---|---|
| 1 | Synthetic Multi-Timescale | Dreamer latent, Flat SAE, Matryoshka-only, Flat-MH, Dense multi-stride, HTS-WM | prefix, level-horizon, factor probes, collapse, boundary, revisitation |
| 2 | MiniHack KeyCorridor-N `N=4..11` | DreamerV3, THICK, DyMoDreamer if easy, Flat-MH, Matryoshka-only, HTS-WM | success vs horizon, hero panel b |
| 3 | Atari 100K full + DMC Visual full | DreamerV3, HarmonyDream, DyMoDreamer, Flat-MH, Matryoshka-only, HTS-WM | guardrail utility, task tables, reliable aggregates |
| 4 | DMC-GB2 | DreamerV3, DyMoDreamer, EAWM, RePo, DreamerPro, SGF-style flat, Flat-MH, Matryoshka-only, HTS-WM | robustness table/figure |

## Immediate Run Commands

Paper-final same-code Breakout, one method and one seed:

```bash
cd /mnt/disk1/backup_user/dat.tt2/xuance

METHODS=dreamer \
SEEDS=0 \
ENV_ID=ALE/Breakout-v5 \
DEVICE=cuda:0 \
WANDB_MODE=online \
PROJECT_NAME=HTS-WM-Paper-Final \
examples/hierarchical_dreamer/run_paper_final_samecode_atari100k.sh
```

Paper-final same-code Breakout, DreamerV3 + HTS-WM for five seeds:

```bash
cd /mnt/disk1/backup_user/dat.tt2/xuance

METHODS=dreamer,htswm \
SEEDS=0,1,2,3,4 \
ENV_ID=ALE/Breakout-v5 \
DEVICE=cuda:0 \
WANDB_MODE=online \
PROJECT_NAME=HTS-WM-Paper-Final \
examples/hierarchical_dreamer/run_paper_final_samecode_atari100k.sh
```

The script defaults to:

```text
CHECKPOINT_RULE=final
TEST_EPISODE=100
RUNNING_STEPS=100000
REPLAY_RATIO=0.125
BATCH_SIZE=16
SEQ_LEN=64
```

## Synthetic Multi-Timescale P0

The Synthetic state-vector dataset spec is now locked and implemented as a
fixed-buffer generator. It does not require Gym/MiniHack/DMC installation.

Generate the default P0 state-vector dataset:

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

Current Synthetic status:

| Component | Status | Notes |
|---|---|---|
| State-vector dynamics | DONE | factors `fast, mid, slow, context, nuisance`; periods `1,4,16,64` |
| State-vector observations | DONE | 44D one-hot concat plus Gaussian noise |
| Labels and IDs | DONE | boundaries, macro/full ids, revisit groups, timestep, episode id |
| Pixel rendering | DONE, P1 | optional `--pixel`; deterministic renderer |
| Reward | PLACEHOLDER | fixed-buffer representation dataset stores zero rewards until a task reward is specified |
| Fixed-buffer evaluator | TODO | metrics listed in `synthetic_multiscale/README.md` |
| Online Gymnasium wrapper | P1 TODO | not a blocker for P0 fixed-buffer experiments |

## Same-Code P0 Baseline Definitions

The updated scope locks these definitions:

| Config name | Status in current code | Required meaning |
|---|---|---|
| `dreamer_anchor` | READY | No auxiliary hierarchy. |
| `flat_single_level_dynamics` | SMOKE ONLY | `L=1` single dynamics head; must not be reported as `flat_mh`. |
| `flat_sae` | READY | One flat TopK sparse code, one decoder, total active budget `48`. |
| `flat_mh` | READY | True flat multi-horizon: one flat latent, six predictors for `[1,2,4,8,16,32]`, no sparsity, no prefixes, no VC. |
| `sgf_style_flat_same_code` | READY | Projector + one-step action-conditioned predictor + VC, no contrastive, no sparsity. |
| `larger_flat_param` | READY FOR BREAKOUT | Search script matches actual HTS add-on params within `2%`; regenerate per env action dim. |
| `larger_flat_flops` | P1 TODO | Match FLOPs/update within `5%`. |
| `recon_only_hierarchy` | READY BY FLAGS | Dense/unconstrained multi-head prefix reconstruction only. |
| `matryoshka_only` | PARTIAL BY FLAGS | Sparse levels + prefix reconstruction + TopK; turn off dynamics, temporal, VC. |
| `dense_multistride_no_sparse` | READY BY FLAGS | Keep levels/prefixes/strides, action-subsequence predictors, temporal loss, and VC; turn off only TopK/L1 and sparsity loss. |
| `dense_multistride_core` | COMPATIBILITY ALIAS | Alias config for `dense_multistride_no_sparse`; do not report as a separate condition. |
| `hts_no_temp` | READY BY FLAGS | Full HTS-WM with temporal loss disabled. |
| `hts_no_vc` | READY BY FLAGS | Full HTS-WM with VC disabled. |
| `hts_full` | READY | Full current HTS-WM. |

Important guardrails:

- Do not report `L=1 + dynamics` as `flat_mh`; call it
  `flat_single_level_dynamics`.
- Do not add contrastive or sparsity losses to `sgf_flat`.
- Do not flatten `dense_multistride` into one concatenated head.
- Do not choose `larger_flat` width by hand; it must be matched from actual
  HTS-WM add-on parameter count.
