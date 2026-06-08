import argparse
import json
import subprocess
from pathlib import Path


LABEL_OUTPUTS = {
    "fig:keycorridor-learning": "figures/keycorridor_learning_curves.pdf",
    "fig:keycorridor-milestones": "figures/keycorridor_milestone_curves.pdf",
    "fig:synthetic-training": "figures/synthetic_training_diagnostics.pdf",
    "fig:hts-ablation-learning": "figures/hts_component_ablation_curves.pdf",
    "fig:atari100k-curves": "figures/atari100k_task_level_curves.pdf",
    "fig:dmc-visual-curves": "figures/dmc_visual_task_level_curves.pdf",
    "fig:dmcgb2-learning": "figures/dmcgb2_robustness_curves.pdf",
}


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main():
    parser = argparse.ArgumentParser("Build paper artifact manifest.")
    parser.add_argument("--artifact-root", default="artifacts")
    parser.add_argument("--paper-tex", default="paper.txt")
    parser.add_argument("--output", default="paper_artifacts/manifest.json")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    output_root = Path(args.output).parent
    paper_tex = Path(args.paper_tex)
    manifest = {
        "code_commit": git_commit(),
        "artifact_root": args.artifact_root,
        "paper_tex": str(paper_tex),
        "paper_tex_found": paper_tex.exists(),
        "confidence_interval_implementation": "seed percentile for smoke; replace with stratified bootstrap/rliable for final",
        "checkpoint_rule": "final for score tables; periodic curves stored separately",
        "labels": [],
        "plot_command": "python examples/hierarchical_dreamer/scripts/build_paper_figures.py --all --artifact-root artifacts --output-root paper_artifacts",
    }
    for label, relpath in LABEL_OUTPUTS.items():
        source_files = [str(path) for path in Path(args.artifact_root).rglob("eval_curve.csv")]
        manifest["labels"].append({
            "latex_label": label,
            "output": str(output_root / relpath),
            "exists": (output_root / relpath).exists(),
            "source_artifacts": source_files[:50],
            "source_artifacts_truncated": len(source_files) > 50,
        })
    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(manifest, indent=2))
    print(f"Wrote manifest to {args.output}")


if __name__ == "__main__":
    main()
