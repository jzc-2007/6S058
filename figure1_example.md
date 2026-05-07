# Figure 1 Example: `museum_bauhaus`

This file captures the four pipeline layers for the `museum_bauhaus` example from the GenPoster run.

## 1) Prompt Text String

**Sample ID:** `museum_bauhaus`  
**Category:** `museum`

**Background (diffusion) prompt**

```text
modern museum exhibition poster background, Bauhaus geometry, red blue yellow blocks, clean gallery lighting, no words, no letters
```

**Requested overlay strings**

```text
BAUHAUS NOW
FORMS IN MOTION
APRIL 12 - JULY 30
```

## 2) Diffusion Background

- Diffusion background: [`outputs/backgrounds/museum_bauhaus.png`](outputs/backgrounds/museum_bauhaus.png)
- Role: no-text background image produced by diffusion

## 3) Overlay Text Layer (Structured)

- Layout JSON: [`outputs/layouts_genposter/museum_bauhaus.json`](outputs/layouts_genposter/museum_bauhaus.json)
- Text blocks (from layout):
  - `BAUHAUS NOW` (title)
  - `FORMS IN MOTION` (subtitle)
  - `APRIL 12 - JULY 30` (meta)

## 4) Final Layer (Composited Poster)

- Final composited output: [`outputs/ours_genposter/museum_bauhaus.png`](outputs/ours_genposter/museum_bauhaus.png)
- The layout metadata points to this file as the rendered source.

## 5) Edited Version (GenPoster Editability)

- Edit before: [`outputs/edit_before_genposter/museum_bauhaus.png`](outputs/edit_before_genposter/museum_bauhaus.png)
- Edit after: [`outputs/edit_after_genposter/museum_bauhaus.png`](outputs/edit_after_genposter/museum_bauhaus.png)
- Edit layout JSON: [`outputs/edit_layouts_genposter/museum_bauhaus.json`](outputs/edit_layouts_genposter/museum_bauhaus.json)

---

Pipeline mapping summary:

1. Prompt text string -> [`prompts/sample_prompts.json`](prompts/sample_prompts.json) (`museum_bauhaus`)
2. Diffusion background -> [`outputs/backgrounds/museum_bauhaus.png`](outputs/backgrounds/museum_bauhaus.png)
3. Overlay text (editable structured layer) -> [`outputs/layouts_genposter/museum_bauhaus.json`](outputs/layouts_genposter/museum_bauhaus.json)
4. Final layer -> [`outputs/ours_genposter/museum_bauhaus.png`](outputs/ours_genposter/museum_bauhaus.png)
5. Edited version -> [`outputs/edit_before_genposter/museum_bauhaus.png`](outputs/edit_before_genposter/museum_bauhaus.png) -> [`outputs/edit_after_genposter/museum_bauhaus.png`](outputs/edit_after_genposter/museum_bauhaus.png) with metadata in [`outputs/edit_layouts_genposter/museum_bauhaus.json`](outputs/edit_layouts_genposter/museum_bauhaus.json)

---

## Qwen New Version Add-on (`qwen_seed_886_tool`)

These links are from the new Qwen-generated run used in the current paper (`qwen_seed_886_tool`), under `outputs_remote/qwen_seed_886/`.

### `album_static`

- Diffusion background: [`outputs_remote/qwen_seed_886/backgrounds/album_static.png`](outputs_remote/qwen_seed_886/backgrounds/album_static.png)
- Final poster: [`outputs_remote/qwen_seed_886/ours_qwen_seed_886_tool/album_static.png`](outputs_remote/qwen_seed_886/ours_qwen_seed_886_tool/album_static.png)
- Layout JSON: [`outputs_remote/qwen_seed_886/layouts_qwen_seed_886_tool/album_static.json`](outputs_remote/qwen_seed_886/layouts_qwen_seed_886_tool/album_static.json)
- Edit before: [`outputs_remote/qwen_seed_886/edit_before_qwen_seed_886_tool/album_static.png`](outputs_remote/qwen_seed_886/edit_before_qwen_seed_886_tool/album_static.png)
- Edit after: [`outputs_remote/qwen_seed_886/edit_after_qwen_seed_886_tool/album_static.png`](outputs_remote/qwen_seed_886/edit_after_qwen_seed_886_tool/album_static.png)
- Edit layout JSON: [`outputs_remote/qwen_seed_886/edit_layouts_qwen_seed_886_tool/album_static.json`](outputs_remote/qwen_seed_886/edit_layouts_qwen_seed_886_tool/album_static.json)

### `concert_neon`

- Diffusion background: [`outputs_remote/qwen_seed_886/backgrounds/concert_neon.png`](outputs_remote/qwen_seed_886/backgrounds/concert_neon.png)
- Final poster: [`outputs_remote/qwen_seed_886/ours_qwen_seed_886_tool/concert_neon.png`](outputs_remote/qwen_seed_886/ours_qwen_seed_886_tool/concert_neon.png)
- Layout planner JSON: [`outputs_remote/qwen_seed_886/layouts_qwen_seed_886_tool/concert_neon.json`](outputs_remote/qwen_seed_886/layouts_qwen_seed_886_tool/concert_neon.json)
- Edit before: [`outputs_remote/qwen_seed_886/edit_before_qwen_seed_886_tool/concert_neon.png`](outputs_remote/qwen_seed_886/edit_before_qwen_seed_886_tool/concert_neon.png)
- Edit after: [`outputs_remote/qwen_seed_886/edit_after_qwen_seed_886_tool/concert_neon.png`](outputs_remote/qwen_seed_886/edit_after_qwen_seed_886_tool/concert_neon.png)
- Edit layout JSON: [`outputs_remote/qwen_seed_886/edit_layouts_qwen_seed_886_tool/concert_neon.json`](outputs_remote/qwen_seed_886/edit_layouts_qwen_seed_886_tool/concert_neon.json)
