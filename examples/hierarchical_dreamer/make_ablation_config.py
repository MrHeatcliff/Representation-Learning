import argparse
import ast
from pathlib import Path

import yaml


def parse_value(raw: str):
    lowered = raw.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered == "null":
        return None
    try:
        return ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return raw


def set_nested(config: dict, dotted_key: str, value):
    target = config
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        target = target.setdefault(part, {})
    target[parts[-1]] = value


def main():
    parser = argparse.ArgumentParser("Generate a Hierarchical Dreamer ablation config.")
    parser.add_argument("--base", required=True, help="Base YAML config.")
    parser.add_argument("--output", required=True, help="Output YAML config.")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Nested override, for example hierarchical_latent.loss_weights.temporal=0.0",
    )
    args = parser.parse_args()

    with Path(args.base).open("r") as file:
        config = yaml.safe_load(file)

    for override in args.set:
        if "=" not in override:
            raise ValueError(f"Override must be KEY=VALUE, got: {override}")
        key, raw_value = override.split("=", 1)
        set_nested(config, key, parse_value(raw_value))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as file:
        yaml.safe_dump(config, file, sort_keys=False)


if __name__ == "__main__":
    main()
