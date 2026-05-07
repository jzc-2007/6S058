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
    cell_w, cell_h = 255, 398
    canvas = Image.new("RGB", (cell_w * len(cols), cell_h * len(items) + 58), "white")
    draw = ImageDraw.Draw(canvas)
    for c, (name, _) in enumerate(cols):
        draw.text((c * cell_w + 14, 16), name, fill=(20, 20, 20), font=font(20))
    for r, item in enumerate(items):
        y = 54 + r * cell_h
        for c, (_, folder) in enumerate(cols):
            canvas.paste(thumb(folder / f"{item['id']}.png"), (c * cell_w + 12, y))
        draw.text((8, y + 352), item["id"], fill=(60, 60, 60), font=font(15))
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.figure)
    print(f"[saved] {args.figure}")


if __name__ == "__main__":
    main()
