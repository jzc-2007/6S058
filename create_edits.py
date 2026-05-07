import argparse
import json
from pathlib import Path

import cv2

from src.rendering import render_text_layer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", type=Path, default=Path("prompts/sample_prompts.json"))
    parser.add_argument("--background-dir", type=Path, default=Path("outputs/backgrounds"))
    parser.add_argument("--before-dir", type=Path, default=Path("outputs/edit_before"))
    parser.add_argument("--after-dir", type=Path, default=Path("outputs/edit_after"))
    parser.add_argument("--before-layout-dir", type=Path, default=Path("outputs/edit_before_layouts"))
    parser.add_argument("--layout-dir", type=Path, default=Path("outputs/edit_layouts"))
    parser.add_argument("--num", type=int, default=6)
    parser.add_argument("--style-map", type=Path, default=None)
    parser.add_argument("--layout-priors", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    items = json.loads(args.prompts.read_text(encoding="utf-8"))[: args.num]
    style_map = {}
    if args.style_map and args.style_map.exists():
        style_map = json.loads(args.style_map.read_text(encoding="utf-8"))
    for item in items:
        sample_id = item["id"]
        bg = cv2.imread(str(args.background_dir / f"{sample_id}.png"), cv2.IMREAD_COLOR)
        if bg is None:
            raise FileNotFoundError(sample_id)
        style_name = style_map.get(sample_id)
        render_text_layer(
            bg,
            item["texts"],
            item.get("category", "poster"),
            args.before_dir / f"{sample_id}.png",
            args.before_layout_dir / f"{sample_id}.json",
            style_name=style_name,
            layout_prior_path=args.layout_priors,
        )
        render_text_layer(
            bg,
            item.get("edit_texts", item["texts"]),
            item.get("category", "poster"),
            args.after_dir / f"{sample_id}.png",
            args.layout_dir / f"{sample_id}.json",
            style_name=style_name,
            layout_prior_path=args.layout_priors,
        )
        print(f"[edit] {sample_id}")


if __name__ == "__main__":
    main()
