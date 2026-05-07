# Layered Typography-Aware Poster Generation

This is a 6.S058 final-project prototype for text-heavy image generation.

The central design choice is to separate image synthesis from typography:

1. Diffusion generates a no-text visual background.
2. A saliency/contrast-aware layout planner places editable text boxes.
3. A deterministic renderer draws the text layer, guaranteeing spelling.
4. OCR/layout/edit-preservation metrics evaluate the outputs.

## Run

```bash
PY=/root/miniconda3/bin/python
$PY -m pip install -r requirements.txt

# quick smoke test, no model download
$PY generate_images.py --provider dummy --mode backgrounds --num 3
$PY render_posters.py --num 3
$PY create_edits.py --num 3
$PY evaluate.py
$PY make_figures.py --num 3

# real diffusion run
$PY generate_images.py --provider diffusers --mode backgrounds --model-id runwayml/stable-diffusion-v1-5
$PY generate_images.py --provider diffusers --mode baselines --model-id runwayml/stable-diffusion-v1-5
$PY render_posters.py
$PY create_edits.py
$PY evaluate.py
$PY make_figures.py
```

## Monitor Long Runs

Long jobs are run in `tmux` with logs:

```bash
tmux attach -t 6s058
tail -f logs/run.log
watch -n 2 nvidia-smi
```

## Outputs

- `outputs/backgrounds`: generated no-text backgrounds.
- `outputs/baselines`: direct T2I images with text requested in the prompt.
- `outputs/ours`: layered poster outputs.
- `outputs/layouts`: structured editable typography JSON.
- `outputs/edit_before`, `outputs/edit_after`: text-edit examples.
- `results/metrics.csv`, `results/summary.json`: quantitative metrics.
- `figures/qualitative_grid.png`, `figures/edit_examples.png`, `figures/pipeline.png`: report figures.
