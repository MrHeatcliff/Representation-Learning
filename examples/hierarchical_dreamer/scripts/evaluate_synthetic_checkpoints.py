import argparse
import csv
import json
from pathlib import Path

import numpy as np


METRICS = [
    "prefix_reconstruction_nrmse",
    "marginal_prefix_gain",
    "level_horizon_prediction_nrmse",
    "predictive_utility_per_active_feature",
    "macro_state_probe_accuracy",
    "factor_probe_fast",
    "factor_probe_mid",
    "factor_probe_slow",
    "factor_probe_context",
    "factor_probe_nuisance",
    "boundary_f1",
    "boundary_detection_delay",
    "false_change_rate",
    "revisit_similarity",
    "activation_ratio",
    "alive_feature_ratio",
    "dead_feature_ratio",
    "minimum_variance",
    "mean_variance",
    "off_diagonal_covariance_norm",
    "topk_utilization_entropy",
    "effective_rank",
]


def load_first_split(dataset_root: Path, split: str):
    manifest = json.loads((dataset_root / "manifest.json").read_text())
    shard = manifest["splits"][split][0]["file"]
    return np.load(dataset_root / shard), manifest


def write_smoke_metrics(dataset_root: Path, output_root: Path, checkpoints: int):
    data, manifest = load_first_split(dataset_root, "val")
    obs = data["obs"].astype(np.float32)
    factors = data["factors"]
    boundaries = data["boundaries"]
    flat_obs = obs.reshape(-1, obs.shape[-1])
    centered = flat_obs - flat_obs.mean(axis=0, keepdims=True)
    variances = centered.var(axis=0)
    boundary_rate = boundaries.reshape(-1, boundaries.shape[-1]).mean(axis=0)
    factor_cardinality = [8, 8, 8, 4, 16]
    factor_acc = [1.0 / value for value in factor_cardinality]

    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / "synthetic_checkpoint_metrics.csv"
    fields = ["checkpoint_index", "global_step", "metric_name", "metric_value", "smoke_metric", "notes"]
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for idx in range(checkpoints):
            progress = idx / max(checkpoints - 1, 1)
            base_rows = {
                "prefix_reconstruction_nrmse": max(0.05, 1.0 - 0.8 * progress),
                "marginal_prefix_gain": 0.1 + 0.2 * progress,
                "level_horizon_prediction_nrmse": max(0.1, 1.2 - 0.7 * progress),
                "predictive_utility_per_active_feature": 0.1 + 0.4 * progress,
                "macro_state_probe_accuracy": 0.25 + 0.6 * progress,
                "factor_probe_fast": factor_acc[0] + (0.95 - factor_acc[0]) * progress,
                "factor_probe_mid": factor_acc[1] + (0.90 - factor_acc[1]) * progress,
                "factor_probe_slow": factor_acc[2] + (0.85 - factor_acc[2]) * progress,
                "factor_probe_context": factor_acc[3] + (0.95 - factor_acc[3]) * progress,
                "factor_probe_nuisance": factor_acc[4] + (0.40 - factor_acc[4]) * progress,
                "boundary_f1": float(boundary_rate[-1]) + 0.5 * progress,
                "boundary_detection_delay": max(0.0, 8.0 * (1.0 - progress)),
                "false_change_rate": max(0.01, float(boundary_rate[0]) * (1.0 - 0.5 * progress)),
                "revisit_similarity": 0.2 + 0.6 * progress,
                "activation_ratio": 48.0 / 192.0,
                "alive_feature_ratio": 0.5 + 0.45 * progress,
                "dead_feature_ratio": 0.5 - 0.45 * progress,
                "minimum_variance": float(variances.min()),
                "mean_variance": float(variances.mean()),
                "off_diagonal_covariance_norm": max(0.01, 0.2 * (1.0 - progress)),
                "topk_utilization_entropy": 0.5 + 0.45 * progress,
                "effective_rank": 8.0 + 24.0 * progress,
            }
            for name in METRICS:
                writer.writerow({
                    "checkpoint_index": idx,
                    "global_step": idx,
                    "metric_name": name,
                    "metric_value": base_rows[name],
                    "smoke_metric": True,
                    "notes": "smoke structural placeholder from synthetic labels; replace with model checkpoint evaluator",
                })
    arrays_path = output_root / "representation_arrays.npz"
    np.savez_compressed(
        arrays_path,
        obs_sample=obs[:8],
        factors_sample=factors[:8],
        boundaries_sample=boundaries[:8],
    )
    print(f"Wrote smoke synthetic metrics to {path}")


def main():
    parser = argparse.ArgumentParser("Evaluate synthetic checkpoints or emit smoke structural metrics.")
    parser.add_argument("--dataset-root", default="data/synthetic_multiscale_state")
    parser.add_argument("--output-root", default="artifacts/smoke_synthetic/synthetic_multiscale/default/hts_full/seed_0")
    parser.add_argument("--num-eval-checkpoints", type=int, default=21)
    parser.add_argument("--smoke", action="store_true", help="Emit smoke metrics without trained checkpoints.")
    args = parser.parse_args()
    if not args.smoke:
        raise SystemExit("Only --smoke is implemented; checkpoint evaluator integration is next.")
    write_smoke_metrics(Path(args.dataset_root), Path(args.output_root), args.num_eval_checkpoints)


if __name__ == "__main__":
    main()
