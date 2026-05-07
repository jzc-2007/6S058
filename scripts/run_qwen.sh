#!/usr/bin/env bash
set -euo pipefail

cd "${PROJECT_DIR:-/root/autodl-tmp/6s058_final_project}"
mkdir -p logs outputs/backgrounds outputs/baselines outputs/ours outputs/layouts \
  outputs/edit_before outputs/edit_after outputs/edit_before_layouts outputs/edit_layouts figures results

PY="${PY:-/root/miniconda3/bin/python}"
MODEL_ID="${MODEL_ID:-Ilus-AI/Qwen-Image-2512-FP8}"
NUM="${NUM:-24}"
STEPS="${STEPS:-20}"
TRUE_CFG_SCALE="${TRUE_CFG_SCALE:-4.0}"
HF_HOME="${HF_HOME:-/root/autodl-tmp/hf_cache}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
STYLE_MAP="${STYLE_MAP:-results/spatial_style_predictions.json}"
LAYOUT_PRIORS="${LAYOUT_PRIORS:-results/spatial_layout_priors.json}"

export HF_HOME HF_ENDPOINT

{
  date
  echo "MODEL_ID=$MODEL_ID NUM=$NUM STEPS=$STEPS TRUE_CFG_SCALE=$TRUE_CFG_SCALE"
  "$PY" generate_images.py --provider diffusers --mode backgrounds \
    --model-id "$MODEL_ID" --num "$NUM" --steps "$STEPS" --true-cfg-scale "$TRUE_CFG_SCALE"
  "$PY" generate_images.py --provider diffusers --mode baselines \
    --model-id "$MODEL_ID" --num "$NUM" --steps "$STEPS" --true-cfg-scale "$TRUE_CFG_SCALE"
  "$PY" render_posters.py --num "$NUM" --style-map "$STYLE_MAP" --layout-priors "$LAYOUT_PRIORS"
  "$PY" create_edits.py --num "$NUM" --style-map "$STYLE_MAP" --layout-priors "$LAYOUT_PRIORS"
  "$PY" evaluate.py
  "$PY" make_figures.py --num 8
  "$PY" make_style_figure.py --num 24 --style-map "$STYLE_MAP" --output-dir outputs/ours --figure figures/spatial_style_grid.png
  date
} 2>&1 | tee logs/qwen_run.log
