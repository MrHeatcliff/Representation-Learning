# Paper Baseline Matrix

This file records the final baseline scope decisions for the current paper
campaign. It is intentionally narrower than the literature review.

## Seed Policy

| Run Type | Seeds |
|---|---|
| Smoke test | one seed |
| Development / screening | `0,1,2` |
| Main Atari / DMC / DMC-GB2 | `0,1,2,3,4` |
| Synthetic mechanism final | at least `0,1,2,3,4`; prefer `0..6` if cheap |
| MiniHack horizon final | `0,1,2,3,4,5,6` |
| Expensive P1 external baseline in main table | `0,1,2,3,4` |
| Expensive P1 external baseline in appendix exploratory | `0,1,2` |

Do not change the seed set between methods within the same table.

## Main Atari Task-Level Columns

Use official SGF only in appendix/external-code references. The main same-code
Atari task-level table should use:

| Column | Required |
|---|---:|
| DreamerV3 | yes |
| HarmonyDream | yes |
| DyMoDreamer | yes |
| Larger-flat parameter-matched | yes |
| Flat multi-horizon | yes |
| Matryoshka-only | yes |
| SGF-style flat same-code | preferred |
| HTS-WM | yes |

Ablations such as no-`L_temp`, no-`L_vc`, dense multi-stride, and recon-only
hierarchy belong in Atari subset/ablation tables first, not full 26-game tables.

## Hero Figure Scope

| Panel | Suite | Metric | Methods |
|---|---|---|---|
| a | DMC Visual full | mean or IQM return | DreamerV3, DyMoDreamer, Flat-MH, Matryoshka-only, HTS-WM |
| a | Atari 100K full | HNS IQM, mean, median | DreamerV3, DyMoDreamer, Flat-MH, Matryoshka-only, HTS-WM |
| a | DMC-GB2 | retained return or relative degradation | DreamerV3, DyMoDreamer, Flat-MH, Matryoshka-only, HTS-WM |
| a | MiniHack memory | success rate | DreamerV3, DyMoDreamer, Flat-MH, Matryoshka-only, HTS-WM |
| b | MiniHack KeyCorridor-N, `N=4..11` | success rate or HTS-WM gain over DreamerV3 | same compact set plus THICK if available |
| c | Synthetic Multi-Timescale | level x horizon prediction heatmap | flat/Matryoshka/Flat-MH/Dense-MH/HTS controls |

Keep HarmonyDream and extra controls in tables/appendix rather than the compact
hero figure.

## Controlled Benchmark Priority

| Priority | Benchmark | Role |
|---:|---|---|
| 1 | Synthetic Multi-Timescale | cheap mechanism validation with ground truth |
| 2 | MiniHack KeyCorridor-N | signature RL horizon sweep |
| 3 | VisualPinPad | sequential subgoal phase memory |
| 4 | Multiworld-Door | visual continuous-control generality check |

## P0 Same-Code Matrix

| Baseline | Synthetic | KeyCorridor | Atari full | DMC full | DMC-GB2 |
|---|---:|---:|---:|---:|---:|
| DreamerV3 | yes | yes | yes | yes | yes |
| Larger-flat parameter-matched | yes | yes | subset | subset | subset |
| Larger-flat FLOPs-matched | yes | yes | subset | subset | subset |
| HarmonyDream | no | optional | yes | subset | optional |
| DyMoDreamer | no | optional | yes | yes | yes |
| SGF-style flat same-code | yes | yes | subset or full | subset | yes |
| Flat SAE | yes | yes | subset | subset | no |
| Flat multi-horizon | yes | yes | subset or full | subset | yes |
| Matryoshka-only | yes | yes | subset or full | subset | yes |
| Recon-only hierarchy | yes | yes | subset | subset | no |
| Dense multi-stride | yes | yes | subset | subset | yes |
| HTS-WM no `L_temp` | yes | yes | subset | subset | yes |
| HTS-WM no `L_vc` | yes | yes | subset | subset | yes |
| HTS-WM | yes | yes | yes | yes | yes |

## Required P1 External Baselines

| Baseline | Suite | Role |
|---|---|---|
| THICK | MiniHack KeyCorridor-N, KeyRoom, VisualPinPad if feasible | adaptive temporal abstraction |
| EAWM | DMC-GB2 | event-aware robustness |
| RePo | DMC-GB2 or matched distractor suite | robust latent representation |
| DreamerPro | DMC-GB2 or DMCR-compatible distractor suite | reconstruction-free robust representation |
| TD-MPC2 | DMC Visual | dense planner/control reference |

## Optional P1 Baselines

| Baseline | Suite | Use If |
|---|---|---|
| TPC | DMC distractor subset | adapter is easy |
| MTS3 | Synthetic fixed-buffer prediction | easy to port to predictive evaluator |
| Denoised MDPs | distractor suite | reproduction code is stable |

## Not In Current Scope

| Baseline / Extension | Reason |
|---|---|
| SPARTAN | no object-centric graph sparsity claim |
| EfficientZero V2 full reproduction | no planner superiority claim |
| CW-VAE on full Atari | not a main utility comparator |
| TPC or Denoised MDPs on every suite | only relevant to distractor suites |
| Full second-backbone campaign | P1 sanity check only |
| Hierarchical planner | P2; no planning-efficiency claim |

