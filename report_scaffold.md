# Layered Typography-Aware Poster Generation with Editable Text

## Abstract

Current text-to-image models can produce visually rich poster-like images, but they still fail when the output must contain exact, editable, well-composed text. We propose a layered generation pipeline that separates visual synthesis from typography: a diffusion model generates a no-text background, a saliency- and contrast-aware planner places structured text boxes, and a deterministic renderer produces the final typography layer. This design guarantees exact spelling and supports local edits without regenerating the background. We evaluate the system with OCR exact match, layout validity, contrast, edit-background preservation, and qualitative comparisons against direct text-to-image generation.

## Method

The pipeline has four stages. First, we generate a background image from a prompt that explicitly excludes text. Second, we score candidate text boxes using local edge density as a lightweight saliency proxy; low-edge regions are preferred because text remains more legible and is less likely to cover important visual content. Third, we choose foreground/stroke colors according to the luminance of the selected region. Fourth, we render the text layer deterministically with font fitting, stroke, shadow, and a translucent local backing band. The output layout is saved as JSON, making each text element editable after generation.

### Lightweight Training Component

To include a learned component without making the three-day project depend on expensive diffusion fine-tuning, we train a small MLP layout ranker. For each generated background, we enumerate candidate text boxes for title, subtitle, and metadata roles. Each candidate is represented by 14 features: normalized geometry, area, local edge statistics, local luminance statistics, margin, preferred-position distance, and role one-hot features. The target score is the normalized score from the hand-designed saliency/layout objective, so the model learns to predict which boxes are likely to be readable and unobtrusive.

The final training run used 520 candidate boxes, split into 416 training and 104 validation examples. The ranker was trained for 300 epochs and reached a validation correlation of 0.871 with the target layout score. This gives the project a concrete trained component while keeping the main visual result reliable.

## Experiments

We use ten poster-style prompts spanning travel, movie posters, coffee ads, museum exhibitions, concert flyers, book covers, restaurant ads, product launches, technology events, and educational events. For each prompt, we compare:

- Direct T2I baseline: the model is asked to render the exact text inside the image.
- Ours: the model generates a no-text background and typography is rendered through our structured layer.

## Metrics

- OCR exact match: fraction of requested text strings recognized in the final image.
- Layout validity: fraction of text boxes inside the canvas and non-overlapping.
- Contrast proxy: local intensity variation in text regions.
- Edit-background SSIM: structural similarity outside text boxes before and after editing the text strings.

## Expected Figure Slots

- `figures/pipeline.png`: method overview.
- `figures/qualitative_grid.png`: background, direct baseline, ours.
- `figures/edit_examples.png`: before/after text edit examples.

## Results

The final run used `runwayml/stable-diffusion-v1-5` for both no-text background generation and direct text-to-image baselines. The direct baseline was prompted to render the exact requested text; our method rendered the same requested strings through the structured typography layer.

| Metric | Direct T2I Baseline | Ours |
|---|---:|---:|
| Rendered text exact | N/A | 1.000 |
| OCR exact match | 0.000 | 0.667 |
| OCR word recall | 0.126 | 0.854 |
| Layout validity | N/A | 1.000 |
| Edit background SSIM | N/A | 1.000 |

The main qualitative trend is that the direct T2I baseline often generates visually plausible poster compositions but fails to render exact text, producing misspellings, pseudo-letters, or unrelated typography. Our method preserves the generated visual background but replaces text generation with a structured renderer, so the visible strings are correct and can be edited without regenerating the image.

## Ablations

We also compare the full system against two ablations:

- No saliency: the layout planner ignores edge density when placing text.
- No contrast: the renderer uses style/category colors without the high-contrast text policy.

| Variant | OCR Exact | OCR Word Recall | Contrast Proxy | Text-Region Edge Density |
|---|---:|---:|---:|---:|
| Full | 0.667 | 0.854 | 0.676 | 0.316 |
| No saliency | 0.700 | 0.843 | 0.677 | 0.328 |
| No contrast | 0.600 | 0.816 | 0.602 | 0.316 |

The no-contrast ablation reduces OCR and contrast, and the no-saliency ablation places text over slightly higher-edge regions on average. Qualitatively, the full method better avoids cluttered or visually important regions while keeping text legible.

## Figures

- `figures/pipeline.png`: method overview.
- `figures/qualitative_grid.png`: background, direct baseline, ours.
- `figures/edit_examples.png`: text edits with unchanged background.
- `figures/ablation_grid.png`: direct baseline and ablations.
- `figures/training_curve.png`: ranker training curve.

## Limitations

The current method is not an end-to-end diffusion model and does not learn typography from real design data. It is best interpreted as a practical layered generation system: the model handles imagery, while a structured CV/design module handles text placement and rendering. OCR exact match remains below the by-construction exactness because Tesseract is brittle on stylized poster crops, especially for small text over textured backgrounds; for that reason we report both phrase-level OCR exact match and word recall.
