#!/usr/bin/env bash
set -uo pipefail

cd "${PROJECT_DIR:-/root/autodl-tmp/6s058_final_project}"
PY="${PY:-/root/miniconda3/bin/python}"
MODEL_ID="${MODEL_ID:-Ilus-AI/Qwen-Image-2512-FP8}"
HF_HOME="${HF_HOME:-/root/autodl-tmp/hf_cache}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
DATA_DIR="${DATA_DIR:-/root/autodl-tmp/genposter_meta}"
DATA_PKL="${DATA_PKL:-$DATA_DIR/0503_raw_offline.pkl}"
RUN_HOURS="${RUN_HOURS:-8}"
STEPS="${STEPS:-20}"
TRUE_CFG_SCALE="${TRUE_CFG_SCALE:-4.0}"

export HF_HOME HF_ENDPOINT
mkdir -p logs results figures "$DATA_DIR"
LOG="logs/overnight_experiments.log"
END_TS=$(( $(date +%s) + RUN_HOURS * 3600 ))

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"
}

run_cmd() {
  log "RUN $*"
  "$@" 2>&1 | tee -a "$LOG"
  local code=${PIPESTATUS[0]}
  log "EXIT code=$code cmd=$*"
  return "$code"
}

download_metadata() {
  if [[ -s "$DATA_PKL" ]]; then
    log "metadata exists: $DATA_PKL"
    return 0
  fi
  log "downloading GenPoster metadata via HF_ENDPOINT=$HF_ENDPOINT"
  "$PY" - <<PY | tee -a "$LOG"
from huggingface_hub import hf_hub_download
path = hf_hub_download(
    repo_id="BruceW91/GenPoster-100K",
    repo_type="dataset",
    filename="0503_raw_offline.pkl",
    local_dir="$DATA_DIR",
)
print(path)
PY
}

render_eval_style_map() {
  local name="$1"
  local style_map="$2"
  local root="${3:-outputs}"
  local bg_dir="$root/backgrounds"
  local baseline_dir="$root/baselines"
  local ours_dir="$root/ours_${name}"
  local layout_dir="$root/layouts_${name}"
  local before_dir="$root/edit_before_${name}"
  local after_dir="$root/edit_after_${name}"
  local before_layout_dir="$root/edit_before_layouts_${name}"
  local edit_layout_dir="$root/edit_layouts_${name}"
  run_cmd "$PY" render_posters.py --num 24 --background-dir "$bg_dir" --output-dir "$ours_dir" --layout-dir "$layout_dir" --style-map "$style_map" --layout-priors results/spatial_layout_priors.json
  run_cmd "$PY" create_edits.py --num 24 --background-dir "$bg_dir" --before-dir "$before_dir" --after-dir "$after_dir" --before-layout-dir "$before_layout_dir" --layout-dir "$edit_layout_dir" --style-map "$style_map" --layout-priors results/spatial_layout_priors.json
  run_cmd "$PY" evaluate.py --ours-dir "$ours_dir" --baseline-dir "$baseline_dir" --layout-dir "$layout_dir" --edit-before-dir "$before_dir" --edit-after-dir "$after_dir" --edit-before-layout-dir "$before_layout_dir" --edit-layout-dir "$edit_layout_dir" --output-csv "results/metrics_${name}.csv"
  run_cmd "$PY" make_style_figure.py --num 24 --style-map "$style_map" --output-dir "$ours_dir" --figure "figures/style_grid_${name}.png"
}

make_fancy_map() {
  "$PY" - <<'PY'
import json
items = json.load(open("prompts/sample_prompts.json"))
style_by_id = {
    "travel_iceland": "classic_travel",
    "movie_solar": "cinematic",
    "coffee_roast": "premium_ad",
    "museum_bauhaus": "bauhaus",
    "concert_neon": "neon",
    "book_orbit": "editorial_serif",
    "food_ramen": "organic_market",
    "tech_summit": "space_age",
    "fitness_launch": "sport",
    "education_space": "space_age",
    "skincare_lumina": "luxury_fashion",
    "jazz_blue_note": "neon",
    "farmers_market": "organic_market",
    "architecture_lecture": "brutalist_type",
    "bike_commute": "classic_travel",
    "tea_ceremony": "script_invite",
    "album_static": "retro_pop",
    "climate_workshop": "clean_corporate",
    "game_launch": "retro_pop",
    "fashion_atelier": "luxury_fashion",
    "photo_exhibit": "bauhaus",
    "charity_run": "sport",
    "boardgame_night": "retro_pop",
    "cookbook_cover": "script_invite",
}
out = {item["id"]: style_by_id.get(item["id"], "playful_social") for item in items}
open("results/fancy_tool_style_predictions.json", "w").write(json.dumps(out, indent=2))
PY
}

log "overnight experiments started; run_hours=$RUN_HOURS model=$MODEL_ID"
download_metadata || log "metadata download failed; continuing with existing style maps"

run_cmd "$PY" -m py_compile train_tool_router.py src/rendering.py src/styles.py evaluate.py create_edits.py render_posters.py make_figures.py make_style_figure.py

if [[ -s "$DATA_PKL" ]]; then
  run_cmd "$PY" train_tool_router.py \
    --metadata-pkl "$DATA_PKL" \
    --max-records 100000 \
    --epochs 120 \
    --hidden-sizes 96,160,256 \
    --seeds 13,23,47,71 \
    --output-style-map results/tool_router_style_predictions.json \
    --summary results/tool_router_summary.json \
    --confusion-csv results/tool_router_confusion.csv \
    --confusion-figure figures/tool_router_confusion.png
  render_eval_style_map "tool_router" "results/tool_router_style_predictions.json" "outputs"
fi

make_fancy_map
render_eval_style_map "fancy_tool" "results/fancy_tool_style_predictions.json" "outputs"
render_eval_style_map "spatial_tool" "results/spatial_style_predictions.json" "outputs"

seed=701
while [[ $(date +%s) -lt $END_TS ]]; do
  root="outputs/qwen_seed_${seed}"
  log "starting Qwen variant seed=$seed root=$root"
  run_cmd "$PY" generate_images.py --provider diffusers --mode backgrounds --output-root "$root" --model-id "$MODEL_ID" --num 24 --steps "$STEPS" --true-cfg-scale "$TRUE_CFG_SCALE" --seed "$seed"
  run_cmd "$PY" generate_images.py --provider diffusers --mode baselines --output-root "$root" --model-id "$MODEL_ID" --num 24 --steps "$STEPS" --true-cfg-scale "$TRUE_CFG_SCALE" --seed "$seed"
  render_eval_style_map "qwen_seed_${seed}_tool" "results/tool_router_style_predictions.json" "$root"
  seed=$(( seed + 37 ))
done

log "overnight experiments finished"
