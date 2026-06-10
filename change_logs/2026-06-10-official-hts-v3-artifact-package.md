# 2026-06-10 Official HTS v3 Artifact Package

Scope: official DreamerV3 HTS port.

## Changes

- Treated `/mnt/disk1/backup_user/dat.tt2/xuance/paper.txt` as canonical manuscript contract.
- Added `flat_partition_dim_matched` as the 15th component-matrix row and official-native HTS variant.
- Exported versioned matrix artifacts:
  - `component_matrix_v3.json`
  - `component_matrix_v3.csv`
  - `component_matrix_v3_parity_report.json`
- Added v3 test reports:
  - `test_report_v3.md`
  - `test_report_v3.csv`
- Added v3 package artifacts for parameter counts, larger-flat search, replay consistency, synthetic smoke, eval smoke, backward trace, and remaining XFAIL.
- Corrected SGF-style action contract in the matrix to use dynamic `action_dim` rather than hard-coded Atari action count.

## Verification

- Component matrix v3 parity passed:
  - JSON rows: `15`
  - CSV rows: `15`
  - config names match
  - schema columns match
- Test report: `13 PASS`, `16 XFAIL`, `0 FAIL`.
- `flat_partition_dim_matched` debug initialization smoke passed.
- Canonical update-rate replay smoke was attempted with expected update/action `0.25`; strict v3 acceptance failed and is recorded honestly.

## Remaining

- Gate A1 remains blocked by canonical replay consistency strict acceptance.
- Gate A2 remains blocked by actual size12m add-on parameter counts, UT-12, RT-01..RT-07, and checkpoint reload/optimizer assertions for every P0 row.
- Gate B remains blocked by real synthetic checkpoint evaluator.
