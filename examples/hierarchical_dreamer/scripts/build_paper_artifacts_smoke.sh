#!/usr/bin/env bash
set -euo pipefail

cd /mnt/disk1/backup_user/dat.tt2/xuance

.venv/bin/python examples/hierarchical_dreamer/scripts/build_paper_tables.py \
  --all \
  --artifact-root artifacts \
  --output-root paper_artifacts

.venv/bin/python examples/hierarchical_dreamer/scripts/build_paper_figures.py \
  --all \
  --artifact-root artifacts \
  --output-root paper_artifacts

.venv/bin/python examples/hierarchical_dreamer/scripts/build_paper_manifest.py \
  --artifact-root artifacts \
  --paper-tex paper.txt \
  --output paper_artifacts/manifest.json
