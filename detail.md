# Final Project Details

This note records what was synced from the remote 5090 run, which outputs are used in the paper, and where to look if we want to swap or hand-pick images.

## Current Paper Version

- Paper source: `submission.tex`
- Compiled PDF: `submission.pdf`
- Main run used in the paper: `qwen_seed_886_tool`
- Main edit-preservation metrics: outside-text SSIM `0.997`, outside-text LPIPS `0.0049`
- Tool-router validation accuracy: `0.789`
- Majority-class baseline: `0.398`

`qwen_seed_886` is the current best paper seed because it has valid layouts across all 24 prompts and the best LPIPS among fully valid Qwen seeds.

## Paper Figures

- `figures/qualitative_grid.png`
  - Paper Figure 2.
  - Each prompt group has three images: no-text Qwen background, direct Qwen T2I baseline, and our layered result.
  - Use this to compare the baseline against our editable typography pipeline.

- `figures/style_grid_mixed_no_overlap.png`
  - Paper Figure 3.
  - 24 final posters in an 8-column by 3-row grid.
  - The first 10 examples are from new held-out demo prompts, and the remaining 14 are from the original 24-prompt run. This avoids prompt overlap with Figure 2 and avoids weak no-text backgrounds from the original run.
  - Use this as the main gallery for style diversity.

- `figures/edit_examples.png`
  - Paper Figure 4.
  - Before/after text edits with the same background preserved.
  - Use this to show the strongest editability claim.

- `figures/tool_router_confusion.png`
  - Supplementary figure.
  - Confusion matrix for the typography-tool router.

- `figures/contact_backgrounds_seed886.png`
  - Contact sheet of generated Qwen backgrounds for seed 886.
  - Use this when checking whether the no-text background still contains accidental text.

- `figures/contact_baselines_seed886.png`
  - Contact sheet of direct Qwen T2I baselines for seed 886.
  - Use this when looking for examples where direct T2I produces pseudo-text, unwanted labels, or non-editable raster typography.

## Seed Gallery Figures

The 8-by-3 style grids for alternative Qwen seeds are in `figures/`:

- `figures/style_grid_qwen_seed_701_tool.png`
- `figures/style_grid_qwen_seed_738_tool.png`
- `figures/style_grid_qwen_seed_775_tool.png`
- `figures/style_grid_qwen_seed_812_tool.png`
- `figures/style_grid_qwen_seed_849_tool.png`
- `figures/style_grid_qwen_seed_886_tool.png`
- `figures/style_grid_qwen_seed_923_tool.png`
- `figures/style_grid_qwen_seed_960_tool.png`
- `figures/style_grid_qwen_seed_997_tool.png`
- `figures/style_grid_qwen_seed_1034_tool.png`
- `figures/style_grid_qwen_seed_1071_tool.png`
- `figures/style_grid_qwen_seed_1108_tool.png`
- `figures/style_grid_qwen_seed_1145_tool.png`
- `figures/style_grid_qwen_seed_1182_tool.png`
- `figures/style_grid_qwen_seed_1219_tool.png`
- `figures/style_grid_qwen_seed_1256_tool.png`
- `figures/style_grid_qwen_seed_1293_tool.png`
- `figures/style_grid_qwen_seed_1330_tool.png`
- `figures/style_grid_qwen_seed_1367_tool.png`
- `figures/style_grid_qwen_seed_1404_tool.png`
- `figures/style_grid_qwen_seed_1441_tool.png`
- `figures/style_grid_qwen_seed_1478_tool.png`
- `figures/style_grid_qwen_seed_1515_tool.png`

If we want to pick a visually better gallery, start from these grids rather than opening all single images one by one. Then check the corresponding metrics in `results/summary_qwen_seed_<seed>_tool.json`.

## Single-Image Outputs

The full synced image outputs are under `outputs_remote/`. These are useful when we want to replace individual examples inside a figure.

For the main seed:

- `outputs_remote/qwen_seed_886/backgrounds/`
  - No-text Qwen background images.

- `outputs_remote/qwen_seed_886/baselines/`
  - Direct Qwen T2I baseline posters.
  - These are generated from prompts that ask Qwen to render the poster text directly.

- `outputs_remote/qwen_seed_886/ours_qwen_seed_886_tool/`
  - Our final layered posters.
  - Qwen generates only the background; the renderer adds structured typography.

- `outputs_remote/qwen_seed_886/layouts_qwen_seed_886_tool/`
  - Layout JSON for the final layered posters.
  - These files include text boxes, styles, fonts, and effects.

- `outputs_remote/qwen_seed_886/edit_before_qwen_seed_886_tool/`
  - Posters before text edits.

- `outputs_remote/qwen_seed_886/edit_after_qwen_seed_886_tool/`
  - Posters after text edits.

- `outputs_remote/qwen_seed_886/edit_before_layouts_qwen_seed_886_tool/`
  - Layout JSON before edits.

- `outputs_remote/qwen_seed_886/edit_layouts_qwen_seed_886_tool/`
  - Layout JSON after edits.

There is also a second synced full run:

- `outputs_remote/qwen_seed_1515/`
  - Same folder structure as seed 886.
  - This is a backup candidate. Its layout validity is lower than seed 886, so it is not the current paper default.

## Metrics

- `results/summary_qwen_seed_886_tool.json`
  - Main summary for the paper seed.

- `results/metrics_qwen_seed_886_tool.csv`
  - Per-prompt metrics for the paper seed.

- `results/summary_qwen_seed_<seed>_tool.json`
  - Summary metrics for each alternative seed.

- `results/metrics_qwen_seed_<seed>_tool.csv`
  - Per-prompt metrics for each alternative seed.

- `results/tool_router_summary.json`
  - Typography-tool router training/validation summary.

- `results/tool_router_confusion.csv`
  - Confusion matrix values used to render the supplementary confusion figure.

## Picking Strategy

For the final paper, prefer examples that satisfy all of the following:

1. The background has no obvious accidental text or watermark.
2. The direct Qwen baseline looks plausible but has visible text problems, pseudo-text, or non-editable typography.
3. Our result has clean readable text, good contrast, and a visibly different style from neighboring examples.
4. For edit examples, the before/after pair changes only the typography layer and leaves the background visually identical.

If replacing Figure 2 examples, pick matching triplets from:

- `outputs_remote/qwen_seed_886/backgrounds/<prompt_id>.png`
- `outputs_remote/qwen_seed_886/baselines/<prompt_id>.png`
- `outputs_remote/qwen_seed_886/ours_qwen_seed_886_tool/<prompt_id>.png`

If replacing Figure 4 examples, pick matching pairs from:

- `outputs_remote/qwen_seed_886/edit_before_qwen_seed_886_tool/<prompt_id>.png`
- `outputs_remote/qwen_seed_886/edit_after_qwen_seed_886_tool/<prompt_id>.png`

Prompt ids include examples such as `concert_neon`, `travel_iceland`, `fashion_atelier`, `tech_summit`, `food_ramen`, `movie_solar`, `museum_bauhaus`, and `book_orbit`.
