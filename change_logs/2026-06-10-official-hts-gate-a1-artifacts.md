# 2026-06-10 Official HTS Gate A1 Artifact Pass

Scope: official DreamerV3 HTS port under `external_baselines/dreamerv3-official`.

## Changes

- Added paper contract note from `paper.txt` because the requested `.tex` file is still missing.
- Replaced ambiguous replay-ratio artifact fields with explicit train-ratio and derived update-rate fields.
- Added replay consistency artifact with prefill/compilation exclusion metadata.
- Fixed CSV artifact writer so existing CSV files receive rows after header creation.
- Added train-loop finalization for checkpoint manifest, final eval placeholder, and replay consistency.
- Integrated artifact writing into official `eval_only` and added an Atari final-eval smoke path.
- Expanded component matrix columns and rows, including `larger_flat_param`, `larger_flat_flops`, `hts_no_hier`, and `hts_no_sdyn`.
- Added numeric add-on parameter estimates for the size12m Atari contract.
- Added named HTS ablation YAML configs for no-temp, no-VC, no-hier, no-sdyn, and dense no-sparsity.
- Clarified training regime names: `joint`, `detach_hts_anchor`, `posthoc_frozen_backbone`, and `two_phase` phase logging.
- Added dynamic one-batch backward trace with gradient norms and one-step parameter deltas.
- Expanded unit-test report format and XFAIL coverage for integration/regression gates.

## Verification

- Python compile check passed for modified official DreamerV3 files.
- Unit report: 7 PASS, 16 XFAIL, 0 FAIL.
- Component matrix regenerated.
- Static and backward debug traces regenerated.
- HTS short Atari artifact smoke passed.
- Eval-only Atari artifact smoke passed with one Pong episode.

## Remaining Gate Status

- Gate A1 is closer but still not fully closed because replay consistency is pending in the very short train smoke with no train metrics.
- Gate A2 is not passed because official-native P0 baselines and parameter-delta tests are still XFAIL.
- Gate B is not passed because the synthetic evaluator remains a smoke placeholder, not a real checkpoint evaluator.
