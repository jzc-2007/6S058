import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", type=Path, default=Path("prompts/sample_prompts.json"))
    parser.add_argument("--figure-dir", type=Path, default=Path("figures"))
    parser.add_argument("--ours-dir", type=Path, default=Path("outputs/ours"))
    parser.add_argument("--edit-before-dir", type=Path, default=Path("outputs/edit_before"))
    parser.add_argument("--edit-after-dir", type=Path, default=Path("outputs/edit_after"))
    parser.add_argument("--num", type=int, default=6)
    return parser.parse_args()


def font(size: int) -> ImageFont.FreeTypeFont:
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            pass
    return ImageFont.load_default()


def thumb(path: Path, size=(260, 390)) -> Image.Image:
    if not path.exists():
        im = Image.new("RGB", size, (235, 235, 235))
        d = ImageDraw.Draw(im)
        d.text((20, size[1] // 2), "missing", fill=(50, 50, 50), font=font(24))
        return im
    im = Image.open(path).convert("RGB")
    im.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, (250, 250, 250))
    canvas.paste(im, ((size[0] - im.width) // 2, (size[1] - im.height) // 2))
    return canvas


def qualitative(items, out: Path, num: int, ours_dir: Path) -> None:
    items = items[:num]
    cell_w, cell_h = 290, 440
    canvas = Image.new("RGB", (cell_w * 3, cell_h * len(items) + 64), "white")
    draw = ImageDraw.Draw(canvas)
    headers = ["Background", "Direct T2I Baseline", "Ours Layered"]
    for i, header in enumerate(headers):
        draw.text((i * cell_w + 18, 18), header, fill=(20, 20, 20), font=font(22))
    for r, item in enumerate(items):
        y = 58 + r * cell_h
        paths = [
            Path("outputs/backgrounds") / f"{item['id']}.png",
            Path("outputs/baselines") / f"{item['id']}.png",
            ours_dir / f"{item['id']}.png",
        ]
        for c, p in enumerate(paths):
            canvas.paste(thumb(p), (c * cell_w + 15, y))
        draw.text((8, y + 398), item["id"], fill=(60, 60, 60), font=font(16))
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)


def edits(items, out: Path, num: int, before_dir: Path, after_dir: Path) -> None:
    items = items[: min(num, 4)]
    cell_w, cell_h = 310, 450
    canvas = Image.new("RGB", (cell_w * 2, cell_h * len(items) + 64), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((18, 18), "Before Edit", fill=(20, 20, 20), font=font(24))
    draw.text((cell_w + 18, 18), "After Text Edit", fill=(20, 20, 20), font=font(24))
    for r, item in enumerate(items):
        y = 58 + r * cell_h
        before = thumb(before_dir / f"{item['id']}.png", (280, 410))
        after = thumb(after_dir / f"{item['id']}.png", (280, 410))
        canvas.paste(before, (15, y))
        canvas.paste(after, (cell_w + 15, y))
        draw.text((8, y + 416), item["id"], fill=(60, 60, 60), font=font(16))
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)


def pipeline(out: Path) -> None:
    canvas = Image.new("RGB", (1400, 420), "white")
    draw = ImageDraw.Draw(canvas)
    boxes = [
        ("Prompt + Text Strings", 50, 130),
        ("No-text Background\nDiffusion", 360, 130),
        ("Saliency + Contrast\nLayout Planner", 670, 130),
        ("Deterministic\nTypography Layer", 980, 130),
    ]
    for label, x, y in boxes:
        draw.rounded_rectangle((x, y, x + 260, y + 130), radius=10, outline=(30, 30, 30), width=3, fill=(246, 247, 248))
        draw.multiline_text((x + 24, y + 36), label, fill=(20, 20, 20), font=font(24), spacing=8)
        if x < 980:
            draw.line((x + 270, y + 65, x + 300, y + 65), fill=(40, 40, 40), width=4)
            draw.polygon([(x + 300, y + 65), (x + 286, y + 57), (x + 286, y + 73)], fill=(40, 40, 40))
    draw.text((60, 325), "Key property: generated image remains editable because text is stored as structured layout JSON.", fill=(20, 20, 20), font=font(28))
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)


def main() -> None:
    args = parse_args()
    items = json.loads(args.prompts.read_text(encoding="utf-8"))
    qualitative(items, args.figure_dir / "qualitative_grid.png", args.num, args.ours_dir)
    edits(items, args.figure_dir / "edit_examples.png", args.num, args.edit_before_dir, args.edit_after_dir)
    pipeline(args.figure_dir / "pipeline.png")
    print(f"[saved] figures in {args.figure_dir}")


if __name__ == "__main__":
    main()
