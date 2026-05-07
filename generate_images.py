import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
from PIL import Image, ImageDraw


NEGATIVE_TEXT = "text, letters, words, logo, watermark, signature, typography, caption"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", type=Path, default=Path("prompts/sample_prompts.json"))
    parser.add_argument("--mode", choices=["backgrounds", "baselines"], default="backgrounds")
    parser.add_argument("--provider", choices=["diffusers", "dummy"], default="diffusers")
    parser.add_argument("--model-id", default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument("--num", type=int, default=None)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance-scale", type=float, default=7.0)
    return parser.parse_args()


def load_prompts(path: Path) -> List[Dict[str, Any]]:
    items = json.loads(path.read_text(encoding="utf-8"))
    return items


def dummy_image(width: int, height: int, seed: int, title: str) -> Image.Image:
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:height, 0:width]
    a = xx / max(1, width - 1)
    b = yy / max(1, height - 1)
    base = np.zeros((height, width, 3), dtype=np.float32)
    palette = rng.integers(20, 235, size=(3, 3))
    base += palette[0] * (1 - a[..., None]) + palette[1] * a[..., None]
    base = base * 0.62 + palette[2] * b[..., None] * 0.38
    for _ in range(10):
        cx, cy = int(rng.integers(0, width)), int(rng.integers(0, height))
        radius = int(rng.integers(height // 12, height // 4))
        color = rng.integers(20, 250, size=3)
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 < radius**2
        base[mask] = 0.72 * base[mask] + 0.28 * color
    image = cv2.GaussianBlur(np.clip(base, 0, 255).astype(np.uint8), (0, 0), 15)
    pil = Image.fromarray(image)
    draw = ImageDraw.Draw(pil)
    draw.rectangle((0, height - 80, width, height), fill=(0, 0, 0))
    draw.text((24, height - 54), f"dummy background: {title}", fill=(255, 255, 255))
    return pil


def load_pipeline(model_id: str):
    import torch
    from diffusers import AutoPipelineForText2Image

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    pipe = AutoPipelineForText2Image.from_pretrained(
        model_id,
        torch_dtype=dtype,
        use_safetensors=True,
        variant="fp16" if "xl" in model_id.lower() else None,
    )
    if torch.cuda.is_available():
        pipe = pipe.to("cuda")
    if hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing()
    return pipe


def main() -> None:
    args = parse_args()
    items = load_prompts(args.prompts)
    if args.num:
        items = items[: args.num]

    out_dir = args.output_root / ("backgrounds" if args.mode == "backgrounds" else "baselines")
    out_dir.mkdir(parents=True, exist_ok=True)

    pipe = None
    if args.provider == "diffusers":
        pipe = load_pipeline(args.model_id)

    for idx, item in enumerate(items):
        sample_id = item["id"]
        width = int(item.get("width", 768))
        height = int(item.get("height", 1152))
        out_path = out_dir / f"{sample_id}.png"
        if out_path.exists():
            print(f"[skip] {out_path}")
            continue

        prompt = item["background_prompt"] if args.mode == "backgrounds" else item["baseline_prompt"]
        if args.provider == "dummy":
            image = dummy_image(width, height, args.seed + idx, sample_id)
        else:
            import torch

            generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(args.seed + idx)
            kwargs = {
                "prompt": prompt,
                "width": width,
                "height": height,
                "num_inference_steps": args.steps,
                "guidance_scale": args.guidance_scale,
                "generator": generator,
            }
            if args.mode == "backgrounds":
                kwargs["negative_prompt"] = NEGATIVE_TEXT
            image = pipe(**kwargs).images[0]

        image.save(out_path)
        print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
