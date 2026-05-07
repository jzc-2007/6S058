import argparse
import json
from pathlib import Path

import cv2

from src.rendering import render_text_layer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", type=Path, default=Path("prompts/sample_prompts.json"))
    parser.add_argument("--background-dir", type=Path, default=Path("outputs/backgrounds"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/ours"))
    parser.add_argument("--layout-dir", type=Path, default=Path("outputs/layouts"))
    parser.add_argument("--num", type=int, default=None)
    parser.add_argument("--no-saliency", action="store_true")
    parser.add_argument("--no-contrast", action="store_true")
    parser.add_argument("--style", type=str, default=None)
    parser.add_argument("--style-map", type=Path, default=None)
    parser.add_argument("--layout-priors", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    items = json.loads(args.prompts.read_text(encoding="utf-8"))
    if args.num:
        items = items[: args.num]
    style_map = {}
    if args.style_map and args.style_map.exists():
        style_map = json.loads(args.style_map.read_text(encoding="utf-8"))
    for item in items:
        sample_id = item["id"]
        bg_path = args.background_dir / f"{sample_id}.png"
        image = cv2.imread(str(bg_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(bg_path)
        render_text_layer(
            image,
            item["texts"],
            item.get("category", "poster"),
            args.output_dir / f"{sample_id}.png",
            args.layout_dir / f"{sample_id}.json",
            saliency_aware=not args.no_saliency,
            contrast_aware=not args.no_contrast,
            style_name=args.style or style_map.get(sample_id),
            layout_prior_path=args.layout_priors,
        )
        print(f"[saved] {args.output_dir / f'{sample_id}.png'}")


if __name__ == "__main__":
    main()
