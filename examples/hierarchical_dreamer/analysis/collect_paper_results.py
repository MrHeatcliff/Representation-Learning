import argparse
import csv
import json
from pathlib import Path

import yaml


SUMMARY_KEYS = {
    "mean_score": [
        "mean_score",
        "eval/episode_reward",
        "Test-Episode-Rewards/Mean-Score",
    ],
    "std_score": [
        "std_score",
        "eval/episode_reward_std",
        "Test-Episode-Rewards/Std-Score",
    ],
    "min_score": [
        "min_score",
        "eval/episode_reward_min",
    ],
    "max_score": [
        "max_score",
        "eval/episode_reward_max",
    ],
    "runtime_seconds": [
        "_runtime",
    ],
    "gradient_step": [
        "step/gradient_step",
        "_step",
    ],
}


def first_present(data, keys, default=""):
    for key in keys:
        if key in data:
            return data[key]
    return default


def load_json(path):
    with Path(path).open("r") as file:
        return json.load(file)


def unwrap_wandb_value(value):
    if isinstance(value, dict) and set(value.keys()) == {"value"}:
        return value["value"]
    return value


def load_wandb_config(path):
    path = Path(path)
    if not path.exists():
        return {}
    with path.open("r") as file:
        raw = yaml.safe_load(file) or {}
    return {key: unwrap_wandb_value(value) for key, value in raw.items()}


def row_from_eval_json(path):
    data = load_json(path)
    return {
        "source_type": "eval_json",
        "source_path": str(path),
        "run_name": data.get("run_name", ""),
        "method": data.get("method", ""),
        "env_id": data.get("env_id", ""),
        "game": data.get("env_id", "").split("/")[-1].replace("-v5", ""),
        "seed": data.get("seed", ""),
        "device": data.get("device", ""),
        "checkpoint_rule": data.get("checkpoint_rule", ""),
        "checkpoint_name": data.get("checkpoint_name", ""),
        "eval_episodes": data.get("eval_episodes", ""),
        "mean_score": data.get("mean_score", ""),
        "std_score": data.get("std_score", ""),
        "min_score": data.get("min_score", ""),
        "max_score": data.get("max_score", ""),
        "runtime_seconds": "",
        "gradient_step": data.get("step", ""),
        "paper_status": "paper_final" if data.get("checkpoint_rule") == "final" else "dev",
        "notes": "",
    }


def row_from_wandb_summary(path):
    data = load_json(path)
    run_dir = Path(path).parents[1]
    config_path = Path(path).with_name("config.yaml")
    config = load_wandb_config(config_path)
    env_id = config.get("env_id", "")
    if not env_id and config.get("game"):
        env_id = str(config.get("game"))
    game = str(env_id).split("/")[-1].replace("-v5", "") if env_id else run_dir.name
    return {
        "source_type": "wandb_summary",
        "source_path": str(path),
        "run_name": config.get("wandb_run_name", run_dir.name),
        "method": config.get("agent", ""),
        "env_id": env_id,
        "game": game,
        "seed": config.get("seed", ""),
        "device": config.get("device", ""),
        "checkpoint_rule": "",
        "checkpoint_name": "",
        "eval_episodes": config.get("test_episode", ""),
        "mean_score": first_present(data, SUMMARY_KEYS["mean_score"]),
        "std_score": first_present(data, SUMMARY_KEYS["std_score"]),
        "min_score": first_present(data, SUMMARY_KEYS["min_score"]),
        "max_score": first_present(data, SUMMARY_KEYS["max_score"]),
        "runtime_seconds": first_present(data, SUMMARY_KEYS["runtime_seconds"]),
        "gradient_step": first_present(data, SUMMARY_KEYS["gradient_step"]),
        "paper_status": "unknown",
        "notes": "fallback row from W&B summary; prefer eval_json when available",
    }


def discover_rows(root):
    root = Path(root)
    rows = []
    seen_summaries = set()
    for path in sorted(root.rglob("eval_results/*_eval.json")):
        rows.append(row_from_eval_json(path))
        wandb_root = next((parent for parent in path.parents if parent.name == "files"), None)
        if wandb_root:
            seen_summaries.add(wandb_root / "wandb-summary.json")

    for path in sorted(root.rglob("wandb-summary.json")):
        if path in seen_summaries:
            continue
        rows.append(row_from_wandb_summary(path))
    return rows


def write_csv(rows, path):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with Path(path).open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows, path):
    headers = [
        "paper_status",
        "method",
        "game",
        "seed",
        "checkpoint_rule",
        "eval_episodes",
        "mean_score",
        "std_score",
        "source_type",
        "source_path",
    ]
    with Path(path).open("w") as file:
        file.write("# Collected Paper Results\n\n")
        if not rows:
            file.write("No result artifacts found.\n")
            return
        file.write("| " + " | ".join(headers) + " |\n")
        file.write("|" + "|".join(["---"] * len(headers)) + "|\n")
        for row in rows:
            values = [str(row.get(header, "")) for header in headers]
            file.write("| " + " | ".join(values) + " |\n")


def main():
    parser = argparse.ArgumentParser("Collect HTS-WM paper result artifacts.")
    parser.add_argument("--root", default=".", help="Repository root or log root to scan.")
    parser.add_argument("--output-dir", default="logs/paper_results", help="Output directory.")
    args = parser.parse_args()

    rows = discover_rows(args.root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(rows, output_dir / "runs.csv")
    write_markdown(rows, output_dir / "runs.md")
    print(f"Collected {len(rows)} rows into {output_dir}")


if __name__ == "__main__":
    main()
