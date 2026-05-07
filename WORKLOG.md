# 6.S058 Final Project Worklog

Last updated: 2026-05-03 10:35 EDT

## Project Goal

Build a 3-day deliverable prototype for **Layered Typography-Aware Poster Generation with Editable Text**.

The project follows the revised practical plan:

1. Generate no-text poster backgrounds with a diffusion model.
2. Generate direct text-to-image baselines that ask the model to render exact text.
3. Render our outputs with a structured typography layer so spelling is exact and edits are stable.
4. Produce quantitative metrics and final report figures.

## Remote Machine

- SSH alias: `seetacloud-6s058-final`
- Project path: `/root/autodl-tmp/6s058_final_project`
- Python: `/root/miniconda3/bin/python`
- GPU: RTX 5090, 32GB VRAM
- PyTorch checked earlier: `2.8.0+cu128`, CUDA available

## Important Commands

Connect:

```bash
ssh seetacloud-6s058-final
cd /root/autodl-tmp/6s058_final_project
```

Monitor GPU:

```bash
watch -n 2 nvidia-smi
```

Monitor long run:

```bash
tmux attach -t 6s058
tail -f /root/autodl-tmp/6s058_final_project/logs/run.log
```

Smoke test:

```bash
cd /root/autodl-tmp/6s058_final_project
bash scripts/run_smoke.sh
```

Diffusion run:

```bash
cd /root/autodl-tmp/6s058_final_project
MODEL_ID=runwayml/stable-diffusion-v1-5 NUM=10 STEPS=30 bash scripts/run_diffusion.sh
```

## Files Created

- `prompts/sample_prompts.json`: 10 poster prompts with background prompts, direct baseline prompts, original text, and edit text.
- `generate_images.py`: dummy or diffusion image generation for backgrounds/baselines.
- `src/layout.py`: edge-density layout planner, contrast-aware color choice, layout validity helpers.
- `src/rendering.py`: deterministic typography renderer.
- `render_posters.py`: render our layered outputs.
- `create_edits.py`: create before/after text edit examples.
- `evaluate.py`: OCR exact match, layout validity, contrast proxy, edit background SSIM.
- `make_figures.py`: qualitative grid, edit grid, and pipeline figure.
- `scripts/run_smoke.sh`: no-model test.
- `scripts/run_diffusion.sh`: end-to-end diffusion run.

## Completed

- Read `draft.md`: it asks for a 3-day plan centered on a high-quality prototype instead of large training.
- Read `intro.pdf`: original introduction framed a typography-aware T2I project with adapter training; current implementation narrows this to a layered editable pipeline.
- Configured remote code in `/root/autodl-tmp/6s058_final_project`.
- Installed Python dependencies from `requirements.txt`.
- Updated scripts to use `/root/miniconda3/bin/python` because non-interactive SSH does not expose `python3` in `PATH`.
- Smoke test passed with dummy backgrounds for 3 samples.
- Installed system OCR binary `tesseract-ocr`.
- Updated evaluation to use crop-based OCR over the structured text boxes; smoke-test OCR exact match for ours is now `1.0`.
- Installed `tmux`.
- First diffusion attempt failed because direct access to `huggingface.co` returned `Network is unreachable`.
- Restarted diffusion using `HF_ENDPOINT=https://hf-mirror.com`; model download is in progress in tmux session `6s058`.
- Noted that smoke-test dummy outputs could contaminate diffusion outputs because existing files were skipped. Patched `scripts/run_diffusion.sh` to clean output folders by default with `CLEAN=1`.
- Mirror download reached about 5.1GB, then failed with `httpx.ConnectTimeout: _ssl.c:983: The handshake operation timed out`. Most weights are cached; only one `.incomplete` file remains. Plan is to restart with longer HF timeouts so it resumes from cache.
- Restart with longer HF timeouts succeeded.
- Generated 10 diffusion backgrounds and 10 direct T2I baselines with `runwayml/stable-diffusion-v1-5`.
- Rendered 10 layered posters and 10 edit pairs.
- Regenerated typography with stronger contrast after initial OCR was too brittle on textured backgrounds.
- Final metrics are in `results/summary.json`.
- Added actual lightweight training: `train_layout_ranker.py` trains an MLP layout ranker over candidate text boxes.
- Training run: 520 examples, 416 train, 104 validation, 300 epochs, final validation correlation `0.8709`; checkpoint saved at `checkpoints/layout_ranker.pt`.
- Added ablations:
  - `outputs/ablations/no_saliency`
  - `outputs/ablations/no_contrast`
  - `figures/ablation_grid.png`
  - `results/metrics_no_saliency.csv`
  - `results/metrics_no_contrast.csv`
  - `results/ablation_layout_summary.json`
- Updated `report_scaffold.md` with training, ablation, and final results sections.
- Prepared multi-style typography support:
  - `src/styles.py` defines eight visual typography styles: `classic_travel`, `cinematic`, `premium_ad`, `bauhaus`, `neon`, `literary`, `tech`, and `sport`.
  - `src/rendering.py` now accepts `style_name` and writes selected style into layout JSON.
  - `render_posters.py` and `create_edits.py` accept `--style-map`.
  - `train_style_lora_adapter.py` trains a low-rank LoRA-style adapter to choose typography style from prompt/category/background features.
  - `make_style_figure.py` builds a style diversity figure after training/rendering.
- User approved style adapter training and it completed:
  - 1280 augmented examples
  - 1024 train / 256 validation
  - 40-dimensional features
  - 8 styles
  - LoRA rank 4
  - 600 epochs
  - final validation accuracy `1.000`
  - checkpoint `checkpoints/style_lora_adapter.pt`
  - predictions `results/style_predictions.json`
- Generated styled results:
  - `outputs/ours_styled`
  - `outputs/edit_before_styled`
  - `outputs/edit_after_styled`
  - `figures/style_grid.png`
  - `figures/style_training_curve.png`
- Wrote and compiled CVPR-style final paper:
  - `submission.tex`
  - `main.bib`
  - `submission.pdf`
  - PDF is 8 pages and compiled without unresolved citations/references.

## Style Adapter Command

```bash
cd /root/autodl-tmp/6s058_final_project
/root/miniconda3/bin/python train_style_lora_adapter.py --epochs 600 --augment 128 --rank 4
/root/miniconda3/bin/python render_posters.py --output-dir outputs/ours_styled --layout-dir outputs/layouts_styled --style-map results/style_predictions.json
/root/miniconda3/bin/python create_edits.py --num 10 --before-dir outputs/edit_before_styled --after-dir outputs/edit_after_styled --layout-dir outputs/edit_layouts_styled --style-map results/style_predictions.json
/root/miniconda3/bin/python make_style_figure.py --num 8
```

## Final Metrics

```json
{
  "rendered_text_exact_ours": 1.0,
  "ocr_exact_ours": 0.6666666666666666,
  "ocr_exact_baseline": 0.0,
  "ocr_word_recall_ours": 0.8544444444444445,
  "ocr_word_recall_baseline": 0.1261904761904762,
  "layout_validity": 1.0,
  "contrast_proxy": 0.6755874415238698,
  "edit_bg_ssim": 1.0000000000000004
}
```

## Current State

No long-running project job is active. Latest enrichment run completed on the remote RTX 5090.

2026-05-03 enrichment:

- Expanded `prompts/sample_prompts.json` from 10 to 24 poster prompts.
- Generated 14 additional no-text backgrounds and 14 additional direct T2I baselines with Stable Diffusion v1.5.
- Retrained layout ranker: 1248 candidates, 450 epochs, final validation MSE 0.2005, final validation correlation 0.8873.
- Retrained LoRA-style typography adapter: 3840 augmented examples, 900 epochs, rank 4, final validation accuracy 1.000.
- Rendered `outputs/ours_styled_v2`, `outputs/edit_before_styled_v2`, `outputs/edit_after_styled_v2`, and `figures/style_grid_v2.png`.
- Fixed `album_static` false-positive black background by neutralizing the background prompt and regenerating only that sample.
- Final 24-prompt metrics in `results/summary_styled_v2.json`:

```json
{
  "rendered_text_exact_ours": 1.0,
  "ocr_exact_ours": 0.7222222222222222,
  "ocr_exact_baseline": 0.0,
  "ocr_word_recall_ours": 0.862962962962963,
  "ocr_word_recall_baseline": 0.07589285714285714,
  "layout_validity": 1.0,
  "contrast_proxy": 0.5692944468723403,
  "edit_bg_ssim": 1.0000000000000002
}
```

Earlier 10-prompt diffusion command:

```bash
cd /root/autodl-tmp/6s058_final_project
HF_ENDPOINT=https://hf-mirror.com HF_HUB_DOWNLOAD_TIMEOUT=120 HF_HUB_ETAG_TIMEOUT=120 CLEAN=1 MODEL_ID=runwayml/stable-diffusion-v1-5 NUM=10 STEPS=30 bash scripts/run_diffusion.sh
```

Final outputs:

- `outputs/backgrounds/*.png`: 10 generated no-text backgrounds
- `outputs/baselines/*.png`: 10 direct text-to-image baselines
- `outputs/ours/*.png`: 10 final layered outputs
- `outputs/layouts/*.json`: editable layout records
- `outputs/edit_before/*.png`, `outputs/edit_after/*.png`: 10 edit pairs
- `results/metrics.csv`
- `results/summary.json`
- `figures/pipeline.png`
- `figures/qualitative_grid.png`
- `figures/edit_examples.png`
- `figures/ablation_grid.png`
- `figures/training_curve.png`
- `checkpoints/layout_ranker.pt`
- `results/training_summary.json`
- `results/summary_no_saliency.json`
- `results/summary_no_contrast.json`
- `checkpoints/style_lora_adapter.pt`
- `results/style_predictions.json`
- `results/style_training_summary.json`
- `figures/style_grid.png`
- `figures/style_training_curve.png`
- `outputs/ours_styled/*.png`
- `outputs/edit_before_styled/*.png`
- `outputs/edit_after_styled/*.png`
- `outputs/ours_styled_v2/*.png`
- `outputs/layouts_styled_v2/*.json`
- `outputs/edit_before_styled_v2/*.png`
- `outputs/edit_after_styled_v2/*.png`
- `outputs/edit_layouts_styled_v2/*.json`
- `results/metrics_styled_v2.csv`
- `results/summary_styled_v2.json`
- `figures/style_grid_v2.png`
- `submission.tex`
- `submission.pdf`

## Recovery Checklist

If the Codex session dies:

1. SSH into the machine with `ssh seetacloud-6s058-final`.
2. `cd /root/autodl-tmp/6s058_final_project`.
3. Read this file: `cat WORKLOG.md`.
4. Check whether a long job is running:

```bash
tmux ls
nvidia-smi
tail -n 80 logs/run.log
tail -n 80 logs/enrich.log
tail -n 80 logs/fix_album.log
```

5. If no job is running and smoke has not passed, run `bash scripts/run_smoke.sh`.
6. If smoke passed but diffusion has not finished, start:

```bash
tmux new -s 6s058
MODEL_ID=runwayml/stable-diffusion-v1-5 NUM=10 STEPS=30 bash scripts/run_diffusion.sh
```

## Known Risks

- `runwayml/stable-diffusion-v1-5` may require model download and network access. If it fails, fallback is dummy backgrounds for pipeline verification, or switch to another public diffusers model.
- OCR depends on `pytesseract` Python package and the system `tesseract` binary. If the binary is missing, install with apt or report OCR as unavailable and use layout/edit metrics plus qualitative figures.
- 50GB data disk is limited; avoid downloading multiple large models at once.

## 2026-05-05 GenPoster Typography Upgrade

User reported three visual issues: black boxes behind text, too few font styles, and repetitive layout formats. I used the new 5090 host alias `seetacloud-6s058-crello` and synced the project to `/root/autodl-tmp/6s058_final_project`.

Dataset decision:
- Checked Crello, Poster100K, PosterCopilot, and GenPoster-100K. Crello has useful font/layout metadata but HF streaming stalled on this machine. Poster100K is useful for text boxes but weaker for font/style metadata. GenPoster-100K exposes layer-level `Text`, `Bounding Box`, `Font`, `FontSize`, `Justification`, and `psd_size`, so it was the best practical source for this upgrade.
- Downloaded GenPoster original metadata only: `BruceW91/GenPoster-100K/0503_raw_offline.pkl`. Avoided image archives.

Training run:
```bash
cd /root/autodl-tmp/6s058_final_project
/root/miniconda3/bin/python train_crello_priors.py \
  --metadata-pkl /root/.cache/huggingface/hub/datasets--BruceW91--GenPoster-100K/snapshots/1c458dec28b1f8f4ae3f6daf8f5d9e3cbbd9067b/0503_raw_offline.pkl \
  --max-records 50000 --layout-epochs 30 --style-epochs 45 --batch-size 2048 --rank 8 \
  --summary results/genposter_pkl_training_summary.json \
  --layout-priors results/genposter_pkl_layout_priors.json \
  --style-map results/genposter_pkl_style_predictions.json \
  --curve figures/genposter_pkl_training_curve.png
```

Training result:
```json
{
  "num_designs": 50000,
  "num_text_designs": 49996,
  "num_layout_examples": 877149,
  "layout_final_val_acc": 0.9506813883781433,
  "style_final_val_acc": 0.43533334136009216
}
```

Important fixes:
- Replaced large black text bands with style-specific local backings: none, underline, side rule, tight panel, and soft panel.
- Expanded from 8 to 13 style families.
- Added real poster layout priors in `results/genposter_pkl_layout_priors.json`.
- Fixed font matching bug where `"Georgia"` matched `NotoSansGeorgian-Bold.ttf`, causing square glyphs for English text. Matching now requires exact/prefix normalized font names.

Final new outputs:
- `outputs/ours_genposter/*.png`
- `outputs/layouts_genposter/*.json`
- `outputs/edit_before_genposter/*.png`
- `outputs/edit_after_genposter/*.png`
- `outputs/edit_layouts_genposter/*.json`
- `figures/genposter_pkl_style_grid.png`
- `figures/genposter_pkl_training_curve.png`
- `results/genposter_pkl_training_summary.json`
- `results/genposter_pkl_layout_priors.json`
- `results/genposter_pkl_style_predictions.json`
- `results/metrics_genposter_pkl.csv`
- `results/summary_genposter_pkl.json`
- `logs/pkl_lora.log`
- `logs/rerender_fontfix.log`

Paper updated and recompiled:
- `submission.tex`
- `main.bib`
- `submission.pdf`

Remote status after completion:
- No `tmux` session running.
- GPU idle, `0 MiB` used.

## 2026-05-05 Full Artistic Typography Upgrade

User asked for more artistic fonts, more diverse layout, more training on the 5090, and editable text. I kept the original text-editing design: every rendered poster still has a matching JSON layout file with text strings, boxes, fonts, colors, strokes, shadows, and style details.

Changes:
- Installed poster/display font packages on the remote machine: Bebas Neue, EB Garamond, Roboto/Roboto Slab/Roboto Condensed, Cabin/Cabin Sketch, Comfortaa, Lobster/Lobster Two, League Spartan, Yusei Magic, Inconsolata, and Fira Code.
- Expanded style families from 13 to 19: `classic_travel`, `cinematic`, `premium_ad`, `bauhaus`, `neon`, `literary`, `tech`, `sport`, `editorial_serif`, `modern_minimal`, `playful_social`, `clean_corporate`, `organic_market`, `luxury_fashion`, `retro_pop`, `handmade_marker`, `script_invite`, `brutalist_type`, and `space_age`.
- Added tracked-letter rendering, per-style shadow opacity, stricter font lookup, and style-specific typography choices.
- Added minimum title/subtitle/meta footprint rules to avoid learned priors shrinking important titles into tiny poster corners.
- Enlarged the layout prior and LoRA style adapter networks, and trained on the full GenPoster-100K metadata release.

Final training command:
```bash
cd /root/autodl-tmp/6s058_final_project
/root/miniconda3/bin/python train_crello_priors.py \
  --metadata-pkl /root/.cache/huggingface/hub/datasets--BruceW91--GenPoster-100K/snapshots/1c458dec28b1f8f4ae3f6daf8f5d9e3cbbd9067b/0503_raw_offline.pkl \
  --max-records 0 \
  --negatives-per-positive 3 \
  --layout-epochs 80 \
  --style-epochs 140 \
  --batch-size 4096 \
  --rank 16 \
  --summary results/art_training_summary.json \
  --layout-priors results/art_layout_priors.json \
  --style-map results/art_style_predictions.json \
  --curve figures/art_training_curve.png
```

Final training result:
```json
{
  "num_designs": 102703,
  "num_text_designs": 102699,
  "num_layout_examples": 2403004,
  "layout_final_val_acc": 0.9686614871025085,
  "style_final_val_acc": 0.6667965054512024,
  "layout_epochs": 80,
  "style_epochs": 140,
  "rank": 16
}
```

Final art outputs:
- `outputs/ours_art/*.png`
- `outputs/layouts_art/*.json`
- `outputs/edit_before_art/*.png`
- `outputs/edit_after_art/*.png`
- `outputs/edit_layouts_art/*.json`
- `figures/art_style_grid.png`
- `figures/qualitative_grid_art.png`
- `figures/edit_examples_art.png`
- `figures/art_training_curve.png`
- `results/art_training_summary.json`
- `results/art_layout_priors.json`
- `results/art_style_predictions.json`
- `results/metrics_art.csv`
- `results/summary_art.json`
- `logs/art_upgrade.log`

Paper:
- `submission.tex` updated to describe the 19-style adapter, full GenPoster training, art font rendering, and editability.
- `submission.pdf` rebuilt locally; final compile succeeded with only a float-page warning.

Current remote status:
- Training complete.
- No tmux session running.
- GPU idle, 0 MiB used.

## 2026-05-06 Spatial/Typography Refinement Run

User reported that the upgraded results are better but still weak on font size, spatial reasoning, and layout quality. Started a second focused run on the 5090 remote.

Code changes:
- `src/layout.py`: added text-length-aware candidate box sizing, template-specific vertical preferences, safe-margin clamping, extra anchor boxes, stronger overlap/near-collision penalties, and larger title/subtitle candidate boxes.
- `src/rendering.py`: raised maximum title/body font size caps and saved rendered metadata (`font_size`, `font_path`, `text_rect`, wrapped text) into layout JSON for edit/audit.
- `train_crello_priors.py`: changed layout prior extraction to filter pathological skinny/vertical real-poster boxes and cluster representative priors per style/role.
- `scripts/layout_audit.py`: added an audit for title font size, text overlaps, and text margins.
- `scripts/run_spatial_upgrade.sh`: added an end-to-end remote run script.

Remote tmux session:
```bash
tmux attach -t spatial_upgrade
```

Log:
```bash
tail -f /root/autodl-tmp/6s058_final_project/logs/spatial_upgrade.log
```

Training/run config:
```bash
/root/miniconda3/bin/python train_crello_priors.py \
  --metadata-pkl /root/.cache/huggingface/hub/datasets--BruceW91--GenPoster-100K/snapshots/1c458dec28b1f8f4ae3f6daf8f5d9e3cbbd9067b/0503_raw_offline.pkl \
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
```

Expected outputs when complete:
- `outputs/ours_spatial/*.png`
- `outputs/layouts_spatial/*.json`
- `outputs/edit_before_spatial/*.png`
- `outputs/edit_after_spatial/*.png`
- `outputs/edit_layouts_spatial/*.json`
- `figures/spatial_style_grid.png`
- `figures/qualitative_grid_spatial.png`
- `figures/edit_examples_spatial.png`
- `figures/spatial_training_curve.png`
- `results/spatial_training_summary.json`
- `results/spatial_layout_priors.json`
- `results/spatial_style_predictions.json`
- `results/layout_audit_spatial.csv`
- `results/layout_audit_spatial.summary.json`
- `logs/spatial_upgrade.log`

Final status:
- Training completed on the 5090.
- Re-rendered once after fixing the remote `src/` rsync path so the final spatial outputs use the new template-aware layout and rendered font metadata.
- Synced final spatial outputs back to local.
- Remote GPU is idle.

Final metrics:
```json
{
  "num_designs": 102703,
  "num_text_designs": 102699,
  "num_layout_examples": 3604506,
  "layout_epochs": 120,
  "style_epochs": 200,
  "layout_final_val_acc": 0.9748111367225647,
  "style_final_val_acc": 0.6665368676185608
}
```

Spatial audit:
```json
{
  "num_layouts": 24,
  "mean_title_font_size": 113.875,
  "min_title_font_size": 69.0,
  "overlap_free_rate": 1.0,
  "mean_min_text_margin": 93.83333333333333
}
```

Final local outputs:
- `outputs/ours_spatial/*.png`
- `outputs/layouts_spatial/*.json`
- `outputs/edit_before_spatial/*.png`
- `outputs/edit_after_spatial/*.png`
- `outputs/edit_layouts_spatial/*.json`
- `figures/spatial_style_grid.png`
- `figures/qualitative_grid_spatial.png`
- `figures/edit_examples_spatial.png`
- `figures/spatial_training_curve.png`
- `results/spatial_training_summary.json`
- `results/spatial_layout_priors.json`
- `results/spatial_style_predictions.json`
- `results/metrics_spatial.csv`
- `results/layout_audit_spatial.csv`
- `results/layout_audit_spatial.summary.json`
- `logs/spatial_upgrade.log`
- `submission.pdf`

Paper:
- `submission.tex` updated to describe spatial refinement and final 3.6M-example training.
- `submission.pdf` rebuilt successfully as an 8-page CVPR-style report.

## 2026-05-05 Final Consistency / Reproducibility Pass

Reviewed the final spatial version for latent issues.

Fixes:
- Added `from __future__ import annotations` to files using Python 3.10 union type syntax so local Python 3.9 can parse them.
- Fixed `scripts/layout_audit.py` local execution under Python 3.9.
- Updated `evaluate.py` so missing optional packages do not crash evaluation:
  - if `pytesseract` is unavailable, OCR metrics become unavailable;
  - if `skimage` is unavailable, edit SSIM becomes unavailable;
  - unavailable values are written as strict JSON `null` instead of non-standard `NaN`.
- Updated OCR crop logic to use rendered `text_rect` metadata when present.
- Generated edit-before/edit-after outputs for all 24 prompts rather than only the first 8.
- Rebuilt `figures/edit_examples_spatial.png`.
- Updated `submission.tex` to avoid mixing old OCR numbers with the final spatial figures. The main table now reports final reproducible spatial metrics: exact rendered text, layout validity, overlap-free rate, title font size, and contrast proxy.

Verification:
```bash
python3 -m py_compile src/layout.py src/rendering.py src/styles.py train_crello_priors.py train_layout_ranker.py evaluate.py create_edits.py render_posters.py scripts/layout_audit.py
bash -n scripts/run_spatial_upgrade.sh
python3 scripts/layout_audit.py --layout-dir outputs/layouts_spatial --output results/layout_audit_spatial.csv
python3 evaluate.py --ours-dir outputs/ours_spatial --layout-dir outputs/layouts_spatial --edit-before-dir outputs/edit_before_spatial --edit-after-dir outputs/edit_after_spatial --edit-layout-dir outputs/edit_layouts_spatial --output-csv results/metrics_spatial.csv
pdflatex -interaction=nonstopmode submission.tex && bibtex submission && pdflatex -interaction=nonstopmode submission.tex && pdflatex -interaction=nonstopmode submission.tex
```

Final local note:
- OCR and SSIM are optional diagnostics on the local machine because `pytesseract` and `skimage` are not installed here. The report no longer depends on those unavailable local metrics for the final spatial claim.
