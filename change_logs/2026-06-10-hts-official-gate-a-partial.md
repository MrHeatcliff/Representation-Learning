# 2026-06-10 HTS Official Gate-A Partial Work

## Summary

Implemented the first Gate-A completion pass for the official DreamerV3 HTS port.
This is not yet paper-final approved because several required components remain
`XFAIL`.

## Implemented

- Exposed canonical HTS config fields:
  - latent anchor metadata
  - level-wise TopK list
  - coarse-to-fine strides
  - per-level normalized hierarchy/dynamics weights
  - temporal contrastive parameters
- Replaced the temporal hinge proxy with a masked InfoNCE-style objective over
  projected coarse code.
- Implemented real far-negative modes:
  - `none`
  - `hard`
  - `soft`
- Added explicit helper functions for unit-testable:
  - level-wise TopK
  - action windows
  - episode/reset masks
  - temporal contrastive masks/loss
  - VICReg loss
- Added partial training regime support:
  - `joint`
  - `frozen` via detached HTS anchor
  - `two_phase` remains config-only / not implemented
- Added component matrix generation.
- Added unit-test runner and one-batch debug trace generator.
- Added replay-ratio semantics note.

## Generated Artifacts

- `external_baselines/dreamerv3-official/paper_artifacts/component_matrix.csv`
- `external_baselines/dreamerv3-official/paper_artifacts/component_matrix.md`
- `external_baselines/dreamerv3-official/paper_artifacts/hts_unit_test_output.json`
- `external_baselines/dreamerv3-official/paper_artifacts/debug_trace_hts_full.json`
- `external_baselines/dreamerv3-official/paper_artifacts/replay_ratio_semantics.md`
- `artifacts/paper_development/synthetic_multiscale_sample/manifest.json`
- `artifacts/paper_development/synthetic_multiscale_sample_eval/synthetic_checkpoint_metrics.csv`

## Verification

- Python compile passed for modified HTS files.
- HTS debug smoke run completed:
  - `/tmp/hts_method_contract_smoke`
- Unit-test runner output:
  - `PASS: 7`
  - `XFAIL: 3`
  - `FAIL: 0`

## Remaining XFAIL

- `UT-12`: true two-phase optimizer groups are not implemented.
- `UT-13`: detached diagnostic decoder is not implemented.
- `UT-15`: official-native P0 baselines are not implemented.

## Paper-Final Status

Gate A is not fully passed. Do not launch paper-final long campaigns from this
state.
