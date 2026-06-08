import argparse
import csv
import shutil
from collections import defaultdict
from pathlib import Path


def read_csvs(root: Path, name: str):
    rows = []
    for path in root.rglob(name):
        with path.open("r", newline="") as file:
            for row in csv.DictReader(file):
                row["_source_file"] = str(path)
                rows.append(row)
    return rows


def write_csv(path: Path, fields: list[str], rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_final_scores(root: Path, output_root: Path):
    rows = read_csvs(root, "final_eval.csv")
    grouped = defaultdict(list)
    context = {}
    for row in rows:
        if row["metric_name"] != "raw_return":
            continue
        key = (row["suite"], row["task"], row["condition"], row["method"], row["seed"])
        grouped[key].append(float(row["metric_value"]))
        context[key] = row
    out = []
    for key, values in sorted(grouped.items()):
        suite, task, condition, method, seed = key
        base = context[key]
        out.append({
            "suite": suite,
            "task": task,
            "condition": condition,
            "method": method,
            "seed": seed,
            "final_return_mean": sum(values) / len(values),
            "final_eval_episodes": len(values),
            "checkpoint_rule": "final",
            "config_hash": base["config_hash"],
            "code_commit": base["code_commit"],
        })
    write_csv(
        output_root / "tables/final_checkpoint_scores.csv",
        ["suite", "task", "condition", "method", "seed", "final_return_mean",
         "final_eval_episodes", "checkpoint_rule", "config_hash", "code_commit"],
        out,
    )
    return out


def build_auc(root: Path, output_root: Path):
    rows = [row for row in read_csvs(root, "eval_curve.csv") if row["metric_name"] in {"raw_return_mean", "hns_mean"}]
    grouped = defaultdict(list)
    for row in rows:
        key = (row["suite"], row["task"], row["condition"], row["method"], row["seed"], row["metric_name"])
        grouped[key].append((float(row["global_step"]), float(row["metric_value"])))
    out = []
    for key, points in sorted(grouped.items()):
        points = sorted(points)
        area = 0.0
        for (x0, y0), (x1, y1) in zip(points, points[1:]):
            area += (x1 - x0) * (y0 + y1) / 2.0
        suite, task, condition, method, seed, metric = key
        horizon = points[-1][0] - points[0][0] if len(points) > 1 else 0.0
        out.append({
            "suite": suite,
            "task": task,
            "condition": condition,
            "method": method,
            "seed": seed,
            "metric_name": metric,
            "auc": area,
            "normalized_auc": area / horizon if horizon > 0 else "",
            "num_points": len(points),
        })
    write_csv(
        output_root / "tables/learning_curve_auc.csv",
        ["suite", "task", "condition", "method", "seed", "metric_name", "auc", "normalized_auc", "num_points"],
        out,
    )
    return out


def build_inventory(root: Path, output_root: Path):
    files = []
    for pattern in ["eval_curve.csv", "final_eval.csv", "representation_metrics.csv", "synthetic_checkpoint_metrics.csv"]:
        for path in root.rglob(pattern):
            files.append({"artifact_type": pattern, "path": str(path), "bytes": path.stat().st_size})
    write_csv(output_root / "tables/curve_generation_inventory.csv", ["artifact_type", "path", "bytes"], files)
    return files


def copy_first_matching(root: Path, output_root: Path, name: str):
    target = output_root / "tables" / name
    matches = list(root.rglob(name))
    if matches:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(matches[0], target)
        return True
    return False


def main():
    parser = argparse.ArgumentParser("Build paper tables from stored artifacts.")
    parser.add_argument("--artifact-root", default="artifacts")
    parser.add_argument("--output-root", default="paper_artifacts")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    artifact_root = Path(args.artifact_root)
    output_root = Path(args.output_root)
    if args.dry_run:
        print(f"Found {len(list(artifact_root.rglob('eval_curve.csv')))} eval_curve.csv files")
        return
    final_rows = build_final_scores(artifact_root, output_root)
    auc_rows = build_auc(artifact_root, output_root)
    inventory = build_inventory(artifact_root, output_root)
    copied_synthetic = copy_first_matching(artifact_root, output_root, "synthetic_checkpoint_metrics.csv")
    for name in ["keycorridor_milestones.csv"]:
        target = output_root / "tables" / name
        if not target.exists():
            write_csv(target, ["status", "notes"], [{"status": "missing_or_smoke", "notes": "not generated yet"}])
    if not copied_synthetic:
        write_csv(
            output_root / "tables/synthetic_checkpoint_metrics.csv",
            ["status", "notes"],
            [{"status": "missing_or_smoke", "notes": "not generated yet"}],
        )
    print(f"Wrote tables: final={len(final_rows)}, auc={len(auc_rows)}, inventory={len(inventory)}")


if __name__ == "__main__":
    main()
