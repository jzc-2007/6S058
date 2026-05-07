#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
METADATA_PKL="${METADATA_PKL:-/root/.cache/huggingface/hub/datasets--BruceW91--GenPoster-100K/snapshots/1c458dec28b1f8f4ae3f6daf8f5d9e3cbbd9067b/0503_raw_offline.pkl}"

"${PYTHON_BIN}" train_crello_priors.py \
  --metadata-pkl "${METADATA_PKL}" \
  --max-records 0 \
  --negatives-per-positive 5 \
  --layout-epochs 120 \
  --style-epochs 200 \
  --batch-size 4096 \
  --rank 24 \
  --summary results/spatial_training_summary.json \
  --layout-priors results/spatial_layout_priors.json \
  --style-map results/spatial_style_predictions.json \
  --curve figures/spatial_training_curve.png

"${PYTHON_BIN}" render_posters.py \
  --style-map results/spatial_style_predictions.json \
  --layout-priors results/spatial_layout_priors.json \
  --output-dir outputs/ours_spatial \
  --layout-dir outputs/layouts_spatial

"${PYTHON_BIN}" create_edits.py \
  --style-map results/spatial_style_predictions.json \
  --layout-priors results/spatial_layout_priors.json \
  --before-dir outputs/edit_before_spatial \
  --after-dir outputs/edit_after_spatial \
  --layout-dir outputs/edit_layouts_spatial \
  --num 8

"${PYTHON_BIN}" evaluate.py \
  --ours-dir outputs/ours_spatial \
  --layout-dir outputs/layouts_spatial \
  --edit-before-dir outputs/edit_before_spatial \
  --edit-after-dir outputs/edit_after_spatial \
  --edit-layout-dir outputs/edit_layouts_spatial \
  --output-csv results/metrics_spatial.csv

"${PYTHON_BIN}" scripts/layout_audit.py \
  --layout-dir outputs/layouts_spatial \
  --output results/layout_audit_spatial.csv

"${PYTHON_BIN}" make_style_figure.py \
  --style-map results/spatial_style_predictions.json \
  --output-dir outputs/ours_spatial \
  --figure figures/spatial_style_grid.png \
  --num 24

"${PYTHON_BIN}" make_figures.py \
  --ours-dir outputs/ours_spatial \
  --edit-before-dir outputs/edit_before_spatial \
  --edit-after-dir outputs/edit_after_spatial \
  --num 8

cp figures/qualitative_grid.png figures/qualitative_grid_spatial.png
cp figures/edit_examples.png figures/edit_examples_spatial.png

echo "[done] spatial upgrade complete"
