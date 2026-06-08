import argparse


def main():
    parser = argparse.ArgumentParser("Unified periodic-evaluation entrypoint.")
    parser.add_argument("--suite", required=True, choices=["atari100k", "synthetic", "keycorridor", "dmc", "dmcgb2"])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.suite == "atari100k":
        print("Atari periodic evaluation is integrated into DreamerV3/HTS-WM Atari entrypoints.")
        print("Use --eval-interval, --intermediate-test-episode, --test-episode, and --checkpoint-rule final.")
        return
    if args.suite == "synthetic":
        print("Use examples/hierarchical_dreamer/scripts/evaluate_synthetic_checkpoints.py --smoke for current P0 smoke.")
        return
    raise SystemExit(f"{args.suite} periodic runner is not implemented yet; setup/dependency work is required.")


if __name__ == "__main__":
    main()
