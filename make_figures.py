import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", type=Path, default=Path("prompts/sample_prompts.json"))
    parser.add_argument("--figure-dir", type=Path, default=Path("figures"))
    parser.add_argument("--background-dir", type=Path, default=Path("outputs/backgrounds"))
    parser.add_argument("--baseline-dir", type=Path, default=Path("outputs/baselines"))
    parser.add_argument("--ours-dir", type=Path, default=Path("outputs/ours"))
    parser.add_argument("--edit-before-dir", type=Path, default=Path("outputs/edit_before"))
    parser.add_argument("--edit-after-dir", type=Path, default=Path("outputs/edit_after"))
    parser.add_argument("--num", type=int, default=8)
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


def qualitative(items, out: Path, num: int, background_dir: Path, baseline_dir: Path, ours_dir: Path) -> None:
    items = items[:num]
    groups_per_row = 2
    variants = [
        ("Background", background_dir),
        ("Direct T2I", baseline_dir),
        ("Ours", ours_dir),
    ]
    cell_w, row_h = 240, 395
    group_gap, header_h = 62, 44
    rows = (len(items) + groups_per_row - 1) // groups_per_row
    group_w = cell_w * len(variants)
    canvas_w = group_w * groups_per_row + group_gap * (groups_per_row - 1)
    canvas = Image.new("RGB", (canvas_w, header_h + row_h * rows), "white")
    draw = ImageDraw.Draw(canvas)
    for g in range(groups_per_row):
        base_x = g * (group_w + group_gap)
        for c, (header, _) in enumerate(variants):
            draw.text((base_x + c * cell_w + 14, 14), header, fill=(20, 20, 20), font=font(18))
    for idx, item in enumerate(items):
        row, group = divmod(idx, groups_per_row)
        base_x = group * (group_w + group_gap)
        y = header_h + row * row_h
        for c, (_, folder) in enumerate(variants):
            canvas.paste(thumb(folder / f"{item['id']}.png", (220, 330)), (base_x + c * cell_w + 10, y))
        draw.text((base_x + 10, y + 338), item["id"], fill=(60, 60, 60), font=font(15))
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)


def edits(items, out: Path, num: int, before_dir: Path, after_dir: Path) -> None:
    items = items[: min(num, 4)]
    groups_per_row = 2
    variants = [("Before", before_dir), ("After", after_dir)]
    cell_w, row_h = 255, 405
    group_gap, header_h = 64, 44
    rows = (len(items) + groups_per_row - 1) // groups_per_row
    group_w = cell_w * len(variants)
    canvas_w = group_w * groups_per_row + group_gap * (groups_per_row - 1)
    canvas = Image.new("RGB", (canvas_w, header_h + row_h * rows), "white")
    draw = ImageDraw.Draw(canvas)
    for g in range(groups_per_row):
        base_x = g * (group_w + group_gap)
        for c, (header, _) in enumerate(variants):
            draw.text((base_x + c * cell_w + 14, 14), header, fill=(20, 20, 20), font=font(18))
    for idx, item in enumerate(items):
        row, group = divmod(idx, groups_per_row)
        base_x = group * (group_w + group_gap)
        y = header_h + row * row_h
        for c, (_, folder) in enumerate(variants):
            canvas.paste(thumb(folder / f"{item['id']}.png", (235, 352)), (base_x + c * cell_w + 10, y))
        draw.text((base_x + 10, y + 360), item["id"], fill=(60, 60, 60), font=font(15))
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
    qualitative(items, args.figure_dir / "qualitative_grid.png", args.num, args.background_dir, args.baseline_dir, args.ours_dir)
    edits(items, args.figure_dir / "edit_examples.png", args.num, args.edit_before_dir, args.edit_after_dir)
    pipeline(args.figure_dir / "pipeline.png")
    print(f"[saved] figures in {args.figure_dir}")


if __name__ == "__main__":
    main()
