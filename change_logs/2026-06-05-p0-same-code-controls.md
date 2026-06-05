# 2026-06-05 P0 Same-Code Controls

## Changed

- Added true same-code P0 control modules to
  `xuance/torch/policies/hierarchical_dreamer.py`:
  - `flat_mh` with horizons `[1, 2, 4, 8, 16, 32]`;
  - `flat_sae` with one TopK flat code and `topk=48`;
  - `sgf_style_flat_same_code` with projector, one-step action-conditioned
    predictor, stop-gradient target, and VICReg.
- Added `hierarchical_latent.control_mode` and flat-control defaults to the
  Atari100K training-regime configs.
- Generated P0 ablation configs under
  `examples/hierarchical_dreamer/config/generated_configs/`.
- Added `search_larger_flat_param.py` to match `larger_flat_param` add-on
  parameters to HTS-WM within tolerance.
- Updated experiment docs and paste-ready commands for the new controls.

## Validation

- `py_compile` passed for modified policy, learner, config-generation scripts,
  and entrypoints.
- Tensor smoke tests passed for `flat_sae`, `flat_mh`, and
  `sgf_style_flat_same_code` modules.
- Breakout `larger_flat_param` search selected width `632`, with analytic add-on
  parameter gap `0.39%` for `actions_dim=4`.

## Remaining

- `larger_flat_flops` is intentionally deferred to P1.
- Paper-final runs still need to be launched and tracked.
