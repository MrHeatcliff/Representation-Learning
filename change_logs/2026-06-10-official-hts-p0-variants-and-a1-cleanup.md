# 2026-06-10 Official HTS P0 Variants And A1 Cleanup

Scope: official DreamerV3 HTS port.

## Implemented

- Corrected component matrix schema:
  - exact row count is now `14`;
  - added source labels for parameter counts;
  - separated target placeholders from actual initialized counts;
  - added decoder/reconstruction fields;
  - added active module/loss fields for ablations.
- Added official-native HTS variants:
  - `flat_sae`
  - `flat_mh`
  - `sgf_style_flat_same_code`
  - `recon_only_hierarchy`
  - `matryoshka_only`
  - `dense_multistride_no_sparse`
  - `larger_flat_param`
- Added YAML presets for the variants above.
- Selected analytical `larger_flat_param` width `2648` with estimated relative gap `0.000122`.
- Converted test output to explicit per-test rows and exported:
  - `paper_artifacts/test_report.csv`
  - `paper_artifacts/test_report.md`
- Ran debug initialization smokes for all P0 variants and named HTS ablations.
- Ran replay-consistency smoke long enough to produce optimizer updates.
- Regenerated dynamic backward trace and verified non-placeholder gradient norms.

## Verification

- Compile check passed.
- Test report: `12 PASS`, `16 XFAIL`, `0 FAIL`.
- Replay consistency smoke:
  - expected updates/action: `0.1`
  - realized updates/action excluding prefill: `0.0790229885`
  - absolute error: `0.0209770115`
  - tolerance: `0.05`
  - status: `pass`
- Final eval smoke remains schema-only, one Pong episode.

## Remaining

- Size12m actual initialized parameter counts for HTS add-ons and larger-flat selected width.
- UT-12 parameter-delta tests.
- RT-01..RT-07 regression tests.
- Periodic evaluation integration.
- Real synthetic checkpoint evaluator.
- Canonical `.tex` manuscript file still missing; contract note uses `paper.txt`.
