import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


def read_eval_rows(root: Path) -> list[dict]:
    rows = []
    for path in root.rglob("eval_curve.csv"):
        with path.open("r", newline="") as file:
            for row in csv.DictReader(file):
                row["_source_file"] = str(path)
                rows.append(row)
    return rows


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)


def aggregate(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        if row.get("metric_name") not in {"raw_return_mean", "hns_mean"}:
            continue
        key = (
            row["experiment_id"],
            row["suite"],
            row["task"],
            row["condition"],
            row["method"],
            row["global_step"],
            row["metric_name"],
        )
        grouped[key].append(float(row["metric_value"]))

    output = []
    for key, values in sorted(grouped.items()):
        experiment_id, suite, task, condition, method, global_step, metric_name = key
        output.append({
            "experiment_id": experiment_id,
            "suite": suite,
            "task": task,
            "condition": condition,
            "method": method,
            "global_step": global_step,
            "metric_name": metric_name,
            "seed_count": len(values),
            "mean": mean(values),
            "median": percentile(values, 0.5),
            "ci95_low": percentile(values, 0.025),
            "ci95_high": percentile(values, 0.975),
            "ci_method": "seed_percentile_smoke; replace with stratified bootstrap for final paper",
        })
    return output


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "experiment_id", "suite", "task", "condition", "method", "global_step",
        "metric_name", "seed_count", "mean", "median", "ci95_low", "ci95_high", "ci_method",
    ]
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser("Aggregate seed-level learning curves.")
    parser.add_argument("--artifact-root", default="artifacts")
    parser.add_argument("--output", default="paper_artifacts/tables/learning_curve_auc.csv")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    rows = aggregate(read_eval_rows(Path(args.artifact_root)))
    if args.dry_run:
        print(f"Would write {len(rows)} aggregate rows to {args.output}")
        return
    write_csv(Path(args.output), rows)
    print(f"Wrote {len(rows)} aggregate rows to {args.output}")


if __name__ == "__main__":
    main()
