# 2026-06-05 Synthetic Multi-Timescale Spec

## Changed

- Added a fixed-buffer Synthetic Multi-Timescale dataset generator at
  `examples/hierarchical_dreamer/synthetic_multiscale/generate_dataset.py`.
- Added Synthetic benchmark README with paste-ready state and pixel dataset
  generation commands.
- Updated paper setup, experiment registry, paste-ready commands, and ablation
  docs with the new locked Synthetic spec and P0 baseline definitions.

## Notes

- Synthetic P0 dataset generation is ready.
- Synthetic evaluator is still missing.
- `flat_mh`, `sgf_flat`, `flat_sae`, `larger_flat`, and true
  `recon_only_hierarchy` still need implementation before paper-final ablations.
- Synthetic reward is currently a zero placeholder because the fixed-buffer
  representation spec did not define a task reward.
