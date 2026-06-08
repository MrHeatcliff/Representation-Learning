import argparse
from pathlib import Path


FIGURES = {
    "fig:keycorridor-learning": "keycorridor_learning_curves.pdf",
    "fig:keycorridor-milestones": "keycorridor_milestone_curves.pdf",
    "fig:synthetic-training": "synthetic_training_diagnostics.pdf",
    "fig:hts-ablation-learning": "hts_component_ablation_curves.pdf",
    "fig:atari100k-curves": "atari100k_task_level_curves.pdf",
    "fig:dmc-visual-curves": "dmc_visual_task_level_curves.pdf",
    "fig:dmcgb2-learning": "dmcgb2_robustness_curves.pdf",
}


def write_minimal_pdf(path: Path, title: str, subtitle: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    text = f"{title}\\n{subtitle}\\nSmoke artifact: replace after full runs."
    stream = f"BT /F1 18 Tf 72 720 Td ({text.replace('(', '[').replace(')', ']')}) Tj ET"
    objects = [
        "1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj",
        "2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj",
        "3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >> endobj",
        "4 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj",
        f"5 0 obj << /Length {len(stream)} >> stream\n{stream}\nendstream endobj",
    ]
    content = "%PDF-1.4\n"
    offsets = [0]
    for obj in objects:
        offsets.append(len(content.encode("utf-8")))
        content += obj + "\n"
    xref_pos = len(content.encode("utf-8"))
    content += f"xref\n0 {len(objects)+1}\n0000000000 65535 f \n"
    for offset in offsets[1:]:
        content += f"{offset:010d} 00000 n \n"
    content += f"trailer << /Size {len(objects)+1} /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF\n"
    path.write_text(content)


def main():
    parser = argparse.ArgumentParser("Build smoke paper figure PDFs from stored artifacts.")
    parser.add_argument("--artifact-root", default="artifacts")
    parser.add_argument("--output-root", default="paper_artifacts")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    output_dir = Path(args.output_root) / "figures"
    for label, filename in FIGURES.items():
        path = output_dir / filename
        if args.dry_run:
            print(f"Would write {label}: {path}")
        else:
            write_minimal_pdf(path, label, f"source root: {args.artifact_root}")
            print(f"Wrote smoke PDF for {label}: {path}")


if __name__ == "__main__":
    main()
