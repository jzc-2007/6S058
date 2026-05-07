import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", type=Path, default=Path("prompts/sample_prompts.json"))
    parser.add_argument("--style-map", type=Path, default=Path("results/style_predictions.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/ours_styled"))
    parser.add_argument("--figure", type=Path, default=Path("figures/style_grid.png"))
    parser.add_argument("--num", type=int, default=24)
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
    style_map = json.loads(args.style_map.read_text(encoding="utf-8")) if args.style_map.exists() else {}
    cols = 8
    cell_w, cell_h = 188, 302
    rows = (len(items) + cols - 1) // cols
    canvas = Image.new("RGB", (cell_w * cols, cell_h * rows), "white")
    draw = ImageDraw.Draw(canvas)
    for idx, item in enumerate(items):
        r, c = divmod(idx, cols)
        x, y = c * cell_w + 9, r * cell_h + 8
        canvas.paste(thumb(args.output_dir / f"{item['id']}.png", (170, 255)), (x, y))
        style = style_map.get(item["id"], "unknown")
        draw.text((x, y + 260), item["id"], fill=(40, 40, 40), font=font(11))
        draw.text((x, y + 276), style, fill=(80, 80, 80), font=font(10))
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.figure)
    print(f"[saved] {args.figure}")


if __name__ == "__main__":
    main()
