import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser("Aggregate Dreamer-style training episode return curves.")
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bins", type=int, default=30)
    parser.add_argument("--max-env-frames", type=int, default=400_000)
    parser.add_argument("--pattern", type=str, default="**/train_episode_returns.csv")
    return parser.parse_args()


def infer_run_fields(path: Path) -> dict[str, str]:
    parts = path.parts
    fields = {"condition": "unknown", "task": "unknown", "seed": "unknown"}
    for part in parts:
        if part.startswith("seed"):
            fields["seed"] = part
    for index, part in enumerate(parts):
        if part in ("p0_smoke", "paper-final", "paper_final") and index + 1 < len(parts):
            fields["condition"] = parts[index + 1]
        if part.endswith("-v5") or part in ("Breakout", "Pong"):
            fields["task"] = part.replace("-v5", "")
    return fields


def read_rows(path: Path) -> list[dict[str, float]]:
    rows = []
    with path.open(newline="") as file:
        for row in csv.DictReader(file):
            rows.append({
                "env_frame": float(row["env_frame"]),
                "episode_return": float(row["episode_return"]),
            })
    return rows


def binned_seed_curve(rows: list[dict[str, float]], bins: np.ndarray) -> np.ndarray:
    values = np.full(len(bins) - 1, np.nan, dtype=np.float64)
    if not rows:
        return values
    xs = np.asarray([row["env_frame"] for row in rows], dtype=np.float64)
    ys = np.asarray([row["episode_return"] for row in rows], dtype=np.float64)
    sums = np.histogram(xs, bins=bins, weights=ys)[0]
    nums = np.histogram(xs, bins=bins)[0]
    valid = nums > 0
    values[valid] = sums[valid] / nums[valid]
    return values


def main():
    args = parse_args()
    bins = np.linspace(0, args.max_env_frames, args.bins + 1)
    grouped = defaultdict(list)
    for path in sorted(args.input_root.glob(args.pattern)):
        fields = infer_run_fields(path)
        key = (fields["condition"], fields["task"])
        grouped[key].append((fields["seed"], binned_seed_curve(read_rows(path), bins)))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "condition",
                "task",
                "bin_index",
                "env_frame_start",
                "env_frame_end",
                "env_frame_mid",
                "seed_count",
                "mean_return",
                "std_return",
            ],
        )
        writer.writeheader()
        for (condition, task), seed_curves in sorted(grouped.items()):
            stacked = np.stack([curve for _, curve in seed_curves], axis=0)
            for index in range(stacked.shape[1]):
                column = stacked[:, index]
                valid = column[~np.isnan(column)]
                writer.writerow({
                    "condition": condition,
                    "task": task,
                    "bin_index": index,
                    "env_frame_start": int(bins[index]),
                    "env_frame_end": int(bins[index + 1]),
                    "env_frame_mid": int((bins[index] + bins[index + 1]) / 2),
                    "seed_count": int(valid.size),
                    "mean_return": "" if valid.size == 0 else float(np.mean(valid)),
                    "std_return": "" if valid.size == 0 else float(np.std(valid)),
                })


if __name__ == "__main__":
    main()
