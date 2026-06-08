import argparse
import csv
import importlib.util
from pathlib import Path


MILESTONES = ["key_observed", "key_picked_up", "door_reached_with_key", "door_opened", "exit_reached"]


def dependency_report():
    return {name: bool(importlib.util.find_spec(name)) for name in ["minihack", "nle", "gym", "gymnasium"]}


def write_empty_schema(output: Path):
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["episode_id", "seed", "task", "corridor_length", "milestone", "reached", "event_step"]
    with output.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for milestone in MILESTONES:
            writer.writerow({
                "episode_id": "",
                "seed": "",
                "task": "",
                "corridor_length": "",
                "milestone": milestone,
                "reached": "",
                "event_step": "",
            })


def main():
    parser = argparse.ArgumentParser("Extract MiniHack KeyCorridor milestone events.")
    parser.add_argument("--output", default="paper_artifacts/tables/keycorridor_milestones.csv")
    parser.add_argument("--check-deps", action="store_true")
    parser.add_argument("--write-empty-schema", action="store_true")
    args = parser.parse_args()
    if args.check_deps:
        for name, available in dependency_report().items():
            print(f"{name}: {available}")
    if args.write_empty_schema:
        write_empty_schema(Path(args.output))
        print(f"Wrote milestone schema to {args.output}")
        return
    deps = dependency_report()
    missing = [name for name, available in deps.items() if not available]
    if missing:
        raise SystemExit(f"Missing MiniHack dependencies: {', '.join(missing)}")
    raise SystemExit("Milestone extraction requires a THICK-compatible KeyCorridor wrapper; not implemented yet.")


if __name__ == "__main__":
    main()
