from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layout-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def overlap_area(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> int:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    return max(0, min(ax1, bx1) - max(ax0, bx0)) * max(0, min(ay1, by1) - max(ay0, by0))


def summarize(path: Path) -> Dict[str, float | str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    width, height = int(payload["width"]), int(payload["height"])
    rects: List[Tuple[int, int, int, int]] = []
    title_font = 0
    min_margin = min(width, height)
    for block in payload.get("rendered_blocks", []):
        rect = tuple(int(v) for v in block.get("text_rect", [0, 0, 0, 0]))
        rects.append(rect)
        if block.get("role") == "title":
            title_font = max(title_font, int(block.get("font_size", 0)))
        min_margin = min(min_margin, rect[0], rect[1], width - rect[2], height - rect[3])

    overlap_px = 0
    for i, rect in enumerate(rects):
        for other in rects[i + 1 :]:
            overlap_px += overlap_area(rect, other)
    title_blocks = [b for b in payload.get("blocks", []) if b.get("role") == "title"]
    title_area = 0.0
    if title_blocks:
        _, _, w, h = title_blocks[0]["bbox"]
        title_area = (w * h) / max(1, width * height)
    return {
        "id": path.stem,
        "style": payload.get("style", ""),
        "num_blocks": len(payload.get("blocks", [])),
        "title_font_size": title_font,
        "title_box_area": round(title_area, 4),
        "text_overlap_px": overlap_px,
        "min_text_margin": min_margin,
    }


def main() -> None:
    args = parse_args()
    rows = [summarize(path) for path in sorted(args.layout_dir.glob("*.json"))]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["id", "style", "num_blocks", "title_font_size", "title_box_area", "text_overlap_px", "min_text_margin"]
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    title_sizes = [float(r["title_font_size"]) for r in rows if float(r["title_font_size"]) > 0]
    overlaps = [float(r["text_overlap_px"]) for r in rows]
    summary = {
        "num_layouts": len(rows),
        "mean_title_font_size": sum(title_sizes) / len(title_sizes) if title_sizes else 0.0,
        "min_title_font_size": min(title_sizes) if title_sizes else 0.0,
        "overlap_free_rate": sum(1 for v in overlaps if v == 0) / len(overlaps) if overlaps else 0.0,
        "mean_min_text_margin": sum(float(r["min_text_margin"]) for r in rows) / len(rows) if rows else 0.0,
    }
    args.output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
