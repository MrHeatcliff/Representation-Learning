# Paper Experiment Registry

This registry maps the updated `paper.txt` experiment plan to concrete result
artifacts. Use LaTeX labels as stable IDs when naming configs, W&B runs, local
folders, and generated figures.

For the actionability split between tasks that are already implementable and
tasks needing user input or external setup, see
`examples/hierarchical_dreamer/PAPER_SETUP_TASKS.md`.

## Paper-Final Rules

The updated paper explicitly requires:

- same-code Dreamer variants use identical env steps, replay ratio, optimizer
  updates, actor-critic, wrappers, and evaluation rules;
- official-code baselines keep native protocols and must be labeled as such;
- headline scores use the final checkpoint unless a benchmark protocol explicitly
  says otherwise;
- Atari 100K final evaluation uses `100` episodes and `5` seeds;
- development runs with fewer seeds or best-checkpoint selection must be labeled
  as development evidence, not paper-final results;
- multi-task results require reliable aggregation: mean, median, IQM, optimality
  gap, probability of improvement, and bootstrap confidence intervals.

Current completed local XuanCe Breakout runs are useful smoke/dev evidence but
are not paper-final because they used best-checkpoint reporting and `3` eval
episodes. The completed SGF run is an official-code external baseline result on
Breakout seed 1 with native final eval over `100` episodes.

## Run Completion Workflow

When a run finishes:

1. Read `wandb-summary.json`, `config.yaml`, `wandb-metadata.json`, and terminal
   `output.log` if present.
2. Add one raw row to `RESULTS_TRACKER.md`.
3. Update the relevant `tab:*` rows below from `TODO` to `PARTIAL` or `DONE`.
4. Record whether the result is `paper-final`, `dev`, `smoke`, or
   `official-code external`.
5. If enough seeds are available, update aggregate rows and figure readiness.
6. Add a short `change_logs/YYYY-MM-DD-*.md` entry.

## Table Registry

| Label | Priority | Paper Role | Required Inputs | Current Status | Where Results Will Be Filled |
|---|---|---|---|---|---|
| `tab:train-setup` | P0 | Shared Dreamer-style training setup | exact batch, seq len, optimizer, replay ratio, checkpoint rule, config hash | PARTIAL | `RESULTS_TRACKER.md`, run configs, W&B metadata |
| `tab:protocol` | P0 | Evaluation-suite protocol | tasks, budgets, repeats, seeds, metrics per suite | PARTIAL | this registry, suite command docs |
| `tab:metrics` | P0 | Metric definitions | implementation/log availability per metric | PARTIAL | this registry, future metric extractors |
| `tab:baselines` | P0 | Core baseline taxonomy | method availability and scope | PARTIAL | baseline docs under `baselines/` |
| `tab:baseline-execution-tiers` | P0/P1 | Required comparator tiers | compatibility and implementation status | PARTIAL | `PAPER_BASELINE_STATUS.md` if added; current baseline docs |
| `tab:main-results` | P0 | Main utility summary | Memory, Motion, Distractor, DMC, Atari, GPU hours | TODO | aggregate tracker after paper-final runs |
| `tab:hero-panel-slots` | P0 | Hero figure suite slots | selected final suites and controls | PARTIAL | frozen slots recorded below; needs runs |
| `tab:prefix` | P0 | Prefix refinement | per-prefix recon error, marginal gain, reward error, active features | TODO | HTS-WM diagnostic logs / future extractor |
| `tab:level-horizon` | P0 | Level-horizon specialization | prediction NRMSE for horizons `1,2,4,8,16,32` | TODO | HTS-WM diagnostic logs / future extractor |
| `tab:temporal-robustness` | P0 | Boundary, revisitation, nuisance robustness | smoothness, boundary F1, delay, false change, revisit sim, nuisance sensitivity | TODO | controlled tasks + paired perturbation logs |
| `tab:ablation-plan` | P0 | Ablation plan | run matrix and status for axes | PARTIAL | `ABLATION_COMMANDS.md` and generated configs |
| `tab:collapse` | P0 | Collapse diagnostics | alive/dead, effective rank, min var, covariance | TODO | HTS-WM diagnostics; current code logs some collapse metrics only if exposed |
| `tab:compute` | P0 | Compute audit | params, extra params, active dims, FLOPs/update, memory, train hours, inference | PARTIAL | W&B `compute/*`, metadata, future FLOPs/latency extractor |
| `tab:cross-domain-protocol` | P0 | Suite protocol audit | exact tasks, inputs, steps, repeat, seeds, replay, model size | TODO | after final suite selection |
| `tab:backbone-reproduction` | P0 | Dreamer anchor reproduction | reference Dreamer, reproduced Dreamer, difference, seeds, GPU, notes | PARTIAL | current Dreamer dev run, future paper-final Dreamer runs |
| `tab:claim-evidence-registry` | P0 | Claim-to-evidence audit | direct evidence, downstream evidence, controls, stress tests | PARTIAL | this registry |
| `tab:experiment-suite-matrix` | P0/P1 | Suite coverage by claim | selected tasks and baseline set per suite | PARTIAL | final scope decisions recorded below; needs environment setup |
| `tab:dreamer-backbone-audits` | P0/P1 | Backbone audit | learning signal, KL, replay, size, data, horizon, encoder update | PARTIAL | some flags exist; many axes need configs |
| `tab:dreamerv3-robustness-audit` | P0/P1 | Dreamer robustness audit | KL, free bits, unimix, clipping, gradient pathways | PARTIAL | some config knobs exist; not all variants implemented |
| `tab:matched-controls` | P0 | Fairness controls | params, active dims, heads, updates, FLOPs/update, replay, train h | PARTIAL | Dreamer/HTS/SGF raw runs; missing matched variants |
| `tab:scaling-grid` | P1 | Compact scaling grid | data, replay, model size, hierarchy budget, seq len, imagination horizon | TODO | future generated configs |
| `tab:nearest-method-matrix` | P0/P1 | Claim-surface comparator matrix | comparator status and compatible suites | PARTIAL | baseline docs + this registry |
| `tab:offline-diagnosis` | P0 | Fixed-buffer representation diagnosis | WM fidelity, reward err, prefix, horizon, boundary, revisit, rank, dead | TODO | offline replay-buffer protocol not implemented |
| `tab:hyper-transfer` | P0 | Hyperparameter transfer audit | axis sweeps, dev tasks, frozen defaults, failure modes | PARTIAL | current config defaults; tuning record incomplete |
| `tab:rollout-fidelity` | P0 | Open-loop prediction error | horizons `1,4,8,16,32`, reward error | TODO | future matched rollout extractor |
| `tab:dmc-task-results` | P1 | DMC task-level results | per-task scores across methods and seeds | TODO | DMC runs not completed |
| `tab:atari-task-results` | P1 | Atari task-level results | all 26 games across methods and seeds, HNS aggregates | PARTIAL | Alien official DreamerV3 size12m has 5-seed online train curve; final `100`-episode eval and other games/methods missing |
| `tab:planner-audit` | P2 | Optional planning audit | planner FLOPs, latency, return | NOT APPLICABLE YET | no hierarchical planner implemented |

## Figure Registry

| Label | Priority | Required Data | Current Status | Output Artifact |
|---|---|---|---|---|
| `fig:overview` | P0 | architecture diagram only | TODO | `figures/overview.*` |
| `fig:hero-results` | P0 | cross-regime summary, horizon sweep, level-horizon heatmap | TODO | `figures/hero_results.*` |
| `fig:horizon-sweep` | P0 | controlled dependency horizon sweep | TODO | `figures/horizon_sweep.*` |
| `fig:rliable-summary` | P1/P0 if Atari central | multi-task scores across seeds | TODO | `figures/rliable_summary.*` |
| `fig:level-horizon` | P0 | level x horizon prediction errors | TODO | `figures/level_horizon.*` |
| `fig:prefix-refinement` | P0 | prefix recon/marginal gains | TODO | `figures/prefix_refinement.*` |
| `fig:spliced-trajectory` | P0 | annotated trajectory with boundaries, revisitation, nuisance | TODO | `figures/spliced_trajectory.*` |
| `fig:nuisance-event` | P0 if robustness claimed | paired clean/perturbed representation changes | TODO | `figures/nuisance_event.*` |
| `fig:collapse-dashboard` | P0 if collapse is central | rank, variance, covariance, alive/dead over checkpoints | TODO | `figures/collapse_dashboard.*` |
| `fig:compute-pareto` | P0 | utility vs GPU h/FLOPs and structure vs active features | TODO | `figures/compute_pareto.*` |
| `fig:loss-interactions` | P1 | raw/weighted losses, grad norms, cosine similarities | TODO | `figures/loss_interactions.*` |
| `fig:prefix-rollouts` | P1 | detached diagnostic decoder rollouts | TODO | `figures/prefix_rollouts.*` |
| `fig:factor-recovery` | P0/P1 controlled | ground-truth factor probes | TODO | `figures/factor_recovery.*` |
| `fig:backbone-audit` | P1 | backbone ablation curves | TODO | `figures/backbone_audit.*` |
| `fig:scaling-transfer` | P1 | scaling and fixed-default transfer curves | TODO | `figures/scaling_transfer.*` |
| `fig:openloop-rollouts` | P1 | matched open-loop predictions | TODO | `figures/openloop_rollouts.*` |
| `fig:task-level-curves` | P0/P1 | per-task learning curves by suite | PARTIAL | Alien official DreamerV3 5-seed curve at `artifacts/paper_development/official_dreamerv3/alien_full26_size12m_5seeds/` |
| `fig:second-backbone` | P1 | second-backbone sanity check | NOT APPLICABLE YET | no second backbone port |

## Master Experiment IDs From Paper

| ID | Priority | Purpose | Current Implementation Status | Result Destination |
|---|---|---|---|---|
| `P0-HERO-01` | P0 | freeze default hierarchy and generate hero figure | PARTIAL: suite slots frozen; paper-final runs missing | `fig:hero-results`, `tab:hero-panel-slots` |
| `P0-MECH-01` | P0 | synthetic factor recovery: flat, Matryoshka, flat-MH, dense-MH, HTS | PARTIAL: Synthetic fixed-buffer generator done; evaluator and true controls missing | `fig:factor-recovery`, `tab:claim-evidence-registry` |
| `P0-MECH-02` | P0 | level-by-horizon heatmap | PARTIAL: HTS losses and Synthetic labels exist; extractor/table not ready | `tab:level-horizon`, `fig:level-horizon` |
| `P0-MECH-03` | P0 | prefix reconstruction and conditional probe gain | PARTIAL: recon losses exist; Synthetic dataset exists; probe/extractor missing | `tab:prefix`, `fig:prefix-refinement` |
| `P0-MECH-04` | P0 | boundary F1, delay, false-change rate | PARTIAL: Synthetic boundary labels exist; evaluator missing | `tab:temporal-robustness`, `fig:spliced-trajectory` |
| `P0-MECH-05` | P0 | revisitation no/hard/soft far negatives | PARTIAL: same-code no/hard/soft sampler implemented with episode masks; revisitation task/extractor missing | `tab:temporal-robustness` |
| `P0-COLL-01` | P0 | no VC, variance-only, covariance-only, smooth-only, global TopK | PARTIAL: config flags exist; result extraction incomplete | `tab:collapse`, `fig:collapse-dashboard` |
| `P0-MEM-01` | P0 | key-door / inventory memory sweep | MISSING: task scripts not wired | `fig:horizon-sweep`, `tab:main-results` |
| `P0-ROB-01` | P0 | video background, moving distractor, camera shake, flicker | MISSING: DMC-GB2/distractor scripts not wired | `fig:nuisance-event`, `tab:temporal-robustness` |
| `P0-BASE-01` | P0 | Dreamer reproduction audit | DEV PARTIAL: Breakout seed 1 only, best-checkpoint/3 eval episodes | `tab:backbone-reproduction` |
| `P0-BASE-02` | P0 | larger-flat parameter/FLOPs matched controls | PARTIAL: `larger_flat_param` implemented/searchable; FLOPs-matched control deferred P1 | `tab:matched-controls` |
| `P0-BASE-03` | P0 | Harmony, DyMo, SGF-flat, flat-SAE, flat-MH, Matryoshka, recon-only | PARTIAL: same-code `flat_mh`, `flat_sae`, `sgf_style_flat_same_code`, `recon_only_hierarchy` implemented; paper-final runs missing | `tab:baselines`, `tab:nearest-method-matrix` |
| `P0-BASE-04` | P0 | T-SAE-style, dense-MH, no-L_hier, no-L_sdyn, no-L_temp, no-L_vc | PARTIAL: `dense_multistride_no_sparse`, `hts_no_hier`, `hts_no_sdyn`, `hts_no_temp`, `hts_no_vc` configs exist; runs pending | `tab:ablation-plan`, `tab:collapse` |
| `P0-OFF-01` | P0 | fixed-buffer diagnosis | PARTIAL: Synthetic dataset generator done; evaluator missing | `tab:offline-diagnosis` |
| `P0-COMP-01` | P0 | params, active dims, FLOPs/update, memory, wall-clock, latency | PARTIAL: params/runtime/memory partially logged; FLOPs/latency missing | `tab:compute`, `fig:compute-pareto` |
| `P1-BASE-01` | P1 | THICK and CW-VAE/MTS3 | NOT STARTED | `tab:nearest-method-matrix` |
| `P1-BASE-02` | P1 | EAWM, TPC, RePo, Denoised MDPs, DreamerPro | PARTIAL: EAWM EADream wrapper/env; others not started | robustness tables/figures |
| `P1-BASE-03` | P1 | SPARTAN object-centric graph task | NOT STARTED | only if graph claims are made |
| `P1-DREAM-01` | P1 | Dreamer robustness-technique audit | PARTIAL: some knobs exist | `tab:dreamerv3-robustness-audit`, `fig:backbone-audit` |
| `P1-SCALE-01` | P1 | data/replay/model/seq/hierarchy scaling | NOT STARTED | `tab:scaling-grid`, `fig:scaling-transfer` |
| `P1-SUITE-01` | P1 | full DMC Visual and Atari 100K | PARTIAL: Atari Alien official DreamerV3 size12m complete for online train-episode curve over seeds `0..4`; final eval and remaining games/methods missing | `tab:dmc-task-results`, `tab:atari-task-results` |
| `P1-SUITE-02` | P1 | Crafter/Craftax and DMLab guardrails | NOT STARTED | optional appendix |
| `P1-ARCH-01` | P1 | second backbone proof of concept | NOT STARTED | `fig:second-backbone` |
| `P2-PLAN-01` | P2 | hierarchical planner | NOT APPLICABLE YET | `tab:planner-audit` |
| `P2-ADAPT-01` | P2 | few-shot mechanism-shift adaptation | NOT STARTED | optional extension |

## Current Result Readiness

| Result | Paper-Final? | Why |
|---|---|---|
| DreamerV3 Breakout seed 1 `cssk65zq` | No, dev only | used best checkpoint and `test_episode=3`; paper asks final checkpoint and `100` eval episodes for Atari |
| HTS-WM Breakout seed 1 `i95tp2se` | No, dev only | used best checkpoint and `test_episode=3`; paper asks final checkpoint and `100` eval episodes for Atari |
| SGF Breakout seed 1 `kcwh4nz5` | Partial official-code external | final eval over `100` episodes, but single seed and native external protocol |
| DreamerV3 official size12m Alien seeds `0..4` | Partial curve/dev | online train episode curve over 5 seeds; no separate final `100`-episode evaluation yet |

## Required Launcher Updates Before Paper-Final Same-Code Runs

| Need | Current State | Required Change |
|---|---|---|
| Atari final eval episodes | DONE: local same-code wrappers now expose `TEST_EPISODE`; paper-final script defaults to `100` | use `run_paper_final_samecode_atari100k.sh` |
| Checkpoint rule | DONE: local DreamerV3 and HTS-WM entrypoints support `--checkpoint-rule best|final`; paper-final script defaults to `final` | use `CHECKPOINT_RULE=final` for paper-final runs |
| Final evaluation artifact | DONE: local DreamerV3 and HTS-WM entrypoints write `<log_dir>/eval_results/{best,final}_eval.json` | parse eval JSON after each run |
| Paper seed set | current completed local runs are only seed `1` | define and run paper seeds, likely `0..4` or explicit user-provided list |
| HNS / rliable aggregates | raw Atari rewards only | add post-processing for HNS, IQM, optimality gap, probability of improvement, bootstrap CI |
| FLOPs / latency | params/runtime partially logged; FLOPs and inference latency missing | add or document a compute extractor before filling `tab:compute` and `fig:compute-pareto` |

## Resolved Scope Decisions

| Decision Point | Resolution |
|---|---|
| Paper-final seed set | Main Atari/DMC/DMC-GB2/same-code baselines use `0,1,2,3,4`; development sweeps use `0,1,2`; MiniHack final uses `0..6`. |
| SGF in main Atari table | Do not include official SGF as a main same-code column. Add same-code `SGF-style flat`; keep official SGF in appendix/external-reference table. |
| Hero panel a | Use DMC Visual full, Atari 100K full, DMC-GB2, and MiniHack memory. |
| Hero panel b | Use MiniHack KeyCorridor-N with `N=4..11`. |
| Hero panel c | Use Synthetic Multi-Timescale level x prediction-horizon heatmap. |
| Controlled benchmark order | Synthetic Multi-Timescale -> MiniHack KeyCorridor-N -> VisualPinPad -> Multiworld-Door. |
| Required P1 external baselines | THICK, EAWM, RePo, DreamerPro, TD-MPC2. |
| Optional P1 external baselines | TPC, MTS3, Denoised MDPs. |
| Skipped for current scope | SPARTAN, full EfficientZero V2 reproduction, full second-backbone campaign, hierarchical planner. |

## Remaining User/External Inputs

- Controlled benchmark setup:
  - Synthetic state-vector fixed-buffer generator is implemented in
    `examples/hierarchical_dreamer/synthetic_multiscale/`; evaluator still
    missing.
  - MiniHack requires a THICK-compatible environment setup and task wrappers.
  - VisualPinPad and Multiworld-Door require adapter decisions after MiniHack.
- DMC and DMC-GB2 require environment setup and task list confirmation.
- Required P1 baselines still need repo/env setup except EAWM, which is partially
  available locally.
- Same-code `SGF-style flat`, `flat_mh`, `flat_sae`, `larger_flat_param`,
  `recon_only_hierarchy`, `matryoshka_only`, `dense_multistride_no_sparse`,
  `hts_no_hier`, `hts_no_sdyn`, `hts_no_temp`, and `hts_no_vc` have code/config paths. They still need paper-final runs and
  result extraction. `larger_flat_flops` and the Synthetic fixed-buffer evaluator
  remain unimplemented.
