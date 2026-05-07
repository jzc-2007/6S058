#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/6s058_final_project
mkdir -p logs
PY="${PY:-/root/miniconda3/bin/python}"
"$PY" generate_images.py --provider dummy --mode backgrounds --num 3 | tee logs/smoke.log
"$PY" render_posters.py --num 3 | tee -a logs/smoke.log
"$PY" create_edits.py --num 3 | tee -a logs/smoke.log
"$PY" evaluate.py | tee -a logs/smoke.log
"$PY" make_figures.py --num 3 | tee -a logs/smoke.log
