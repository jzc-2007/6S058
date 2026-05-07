#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/6s058_final_project
mkdir -p logs

MODEL_ID="${MODEL_ID:-runwayml/stable-diffusion-v1-5}"
NUM="${NUM:-10}"
STEPS="${STEPS:-30}"
PY="${PY:-/root/miniconda3/bin/python}"
CLEAN="${CLEAN:-1}"

if [ "$CLEAN" = "1" ]; then
  rm -rf outputs/backgrounds outputs/baselines outputs/ours outputs/layouts \
    outputs/edit_before outputs/edit_after outputs/edit_layouts figures results
  mkdir -p outputs/backgrounds outputs/baselines outputs/ours outputs/layouts \
    outputs/edit_before outputs/edit_after outputs/edit_layouts figures results logs
fi

{
  date
  echo "MODEL_ID=$MODEL_ID NUM=$NUM STEPS=$STEPS"
  "$PY" generate_images.py --provider diffusers --mode backgrounds --model-id "$MODEL_ID" --num "$NUM" --steps "$STEPS"
  "$PY" generate_images.py --provider diffusers --mode baselines --model-id "$MODEL_ID" --num "$NUM" --steps "$STEPS"
  "$PY" render_posters.py --num "$NUM"
  "$PY" create_edits.py --num 6
  "$PY" evaluate.py
  "$PY" make_figures.py --num 6
  date
} 2>&1 | tee logs/run.log
