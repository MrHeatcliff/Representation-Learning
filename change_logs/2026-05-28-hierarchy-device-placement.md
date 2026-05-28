# 2026-05-28 - Hierarchy Device Placement

## Summary

Fixed a CUDA/CPU mismatch in Hierarchical Dreamer.

## Issue

The Dreamer recurrent state `h_t` was on CUDA during training, while hierarchy-specific modules created in `HierarchicalDreamerPolicy` stayed on CPU.

This caused errors such as:

```text
RuntimeError: Expected all tensors to be on the same device, but got mat1 is on cuda:1, different from other tensors on cpu
```

## Fix

Moved all hierarchy-specific modules to `self.device` after construction:

- sparse latent heads
- sparse latent readout
- nested reconstruction decoders
- multi-stride sparse dynamics predictors
- temporal contrastive/VICReg projector
