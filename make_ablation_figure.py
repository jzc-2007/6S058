import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", type=Path, default=Path("prompts/sample_prompts.json"))
    parser.add_argument("--figure", type=Path, default=Path("figures/ablation_grid.png"))
    parser.add_argument("--num", type=int, default=4)
    return parser.parse_args()


def font(size: int) -> ImageFont.FreeTypeFont:
    for path in ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            pass
    return ImageFont.load_default()


def thumb(path: Path, size=(230, 345)) -> Image.Image:
    im = Image.open(path).convert("RGB") if path.exists() else Image.new("RGB", size, (235, 235, 235))
    im.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, "white")
    canvas.paste(im, ((size[0] - im.width) // 2, (size[1] - im.height) // 2))
    return canvas


def main() -> None:
    args = parse_args()
    items = json.loads(args.prompts.read_text(encoding="utf-8"))[: args.num]
    cols = [
        ("Direct T2I", Path("outputs/baselines")),
        ("No Saliency", Path("outputs/ablations/no_saliency")),
        ("No Contrast", Path("outputs/ablations/no_contrast")),
        ("Full", Path("outputs/ours")),
    ]
    groups_per_row = 2
    cell_w, row_h = 178, 300
    group_gap, header_h = 58, 38
    rows = (len(items) + groups_per_row - 1) // groups_per_row
    group_w = cell_w * len(cols)
    canvas_w = group_w * groups_per_row + group_gap * (groups_per_row - 1)
    canvas = Image.new("RGB", (canvas_w, header_h + row_h * rows), "white")
    draw = ImageDraw.Draw(canvas)
    for g in range(groups_per_row):
        base_x = g * (group_w + group_gap)
        for c, (_, folder) in enumerate(cols):
            draw.text((base_x + c * cell_w + 8, 13), cols[c][0], fill=(20, 20, 20), font=font(13))
    for idx, item in enumerate(items):
        row, group = divmod(idx, groups_per_row)
        base_x = group * (group_w + group_gap)
        y = header_h + row * row_h
        for c, (_, folder) in enumerate(cols):
            canvas.paste(thumb(folder / f"{item['id']}.png", (162, 243)), (base_x + c * cell_w + 8, y))
        draw.text((base_x + 8, y + 250), item["id"], fill=(60, 60, 60), font=font(12))
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.figure)
    print(f"[saved] {args.figure}")


if __name__ == "__main__":
    main()
