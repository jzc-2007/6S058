from __future__ import annotations

from pathlib import Path
import json
from typing import Iterable, Sequence

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from .layout import TextBlock, plan_layout, save_layout
from .styles import TypographyStyle, get_style


FONT_DIRS = [
    Path("/usr/share/fonts"),
    Path("/usr/local/share/fonts"),
    Path("/System/Library/Fonts"),
    Path("/System/Library/Fonts/Supplemental"),
    Path("/Library/Fonts"),
]


def _font_inventory() -> list[Path]:
    fonts: list[Path] = []
    for root in FONT_DIRS:
        if root.exists():
            fonts.extend(root.rglob("*.ttf"))
            fonts.extend(root.rglob("*.otf"))
    return fonts


def _normalize_font_name(value: str) -> str:
    return "".join(ch.lower() for ch in value if ch.isalnum())


def _font_query_keys(name: str) -> list[str]:
    key = _normalize_font_name(name)
    weight_words = ["regular", "bold", "black", "medium", "semibold", "light", "thin", "extrabold", "condensed"]
    keys = [key]
    for word in weight_words:
        if key.endswith(word):
            keys.append(key[: -len(word)])
    return [k for k in keys if k]


def find_font(preferred: str | None = None, names: Iterable[str] = ()) -> Path:
    candidates: list[Path] = []
    if preferred:
        candidates.append(Path(preferred))
    fonts = _font_inventory()
    by_name = {str(path): path for path in fonts}
    normalized = {str(path): _normalize_font_name(path.stem + " " + path.parent.name) for path in fonts}
    for name in names:
        for key in _font_query_keys(name):
            found = False
            for raw, normed in normalized.items():
                if normed == key or normed.startswith(key):
                    candidates.append(by_name[raw])
                    found = True
                    break
            if found:
                break
    candidates.extend(
        [
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed-Bold.ttf"),
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
            Path("/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf"),
            Path("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"),
            Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf"),
            Path("/System/Library/Fonts/Supplemental/Georgia Bold.ttf"),
        ]
    )
    for path in candidates:
        if path.exists() and path.suffix.lower() in {".ttf", ".otf"}:
            return path
    raise FileNotFoundError("No usable TrueType font found.")


def _wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, max_width: int) -> str:
    words = text.split()
    if not words:
        return text
    lines = []
    current = words[0]
    for word in words[1:]:
        trial = current + " " + word
        if draw.textbbox((0, 0), trial, font=font)[2] <= max_width:
            current = trial
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return "\n".join(lines)


def _text_width(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, tracking: int) -> int:
    if tracking <= 0 or len(text) <= 1:
        bbox = draw.textbbox((0, 0), text, font=font)
        return bbox[2] - bbox[0]
    return sum(draw.textbbox((0, 0), ch, font=font)[2] for ch in text) + tracking * (len(text) - 1)


def _wrap_text_tracked(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, max_width: int, tracking: int) -> str:
    words = text.split()
    if not words:
        return text
    lines = []
    current = words[0]
    for word in words[1:]:
        trial = current + " " + word
        if _text_width(draw, trial, font, tracking) <= max_width:
            current = trial
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return "\n".join(lines)


def _tracked_multiline_bbox(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont,
    spacing: int,
    tracking: int,
    stroke_width: int = 0,
) -> tuple[int, int, int, int]:
    lines = text.split("\n") or [text]
    widths = [_text_width(draw, line, font, tracking) + stroke_width * 2 for line in lines]
    line_boxes = [draw.textbbox((0, 0), line or " ", font=font, stroke_width=stroke_width) for line in lines]
    heights = [box[3] - box[1] for box in line_boxes]
    return (0, 0, max(widths, default=0), sum(heights) + spacing * max(0, len(lines) - 1))


def _draw_tracked_multiline(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    font: ImageFont.FreeTypeFont,
    fill: tuple[int, int, int, int],
    spacing: int,
    align: str,
    tracking: int,
    stroke_width: int = 0,
    stroke_fill: tuple[int, int, int, int] | None = None,
) -> None:
    if tracking <= 0:
        draw.multiline_text(
            xy,
            text,
            font=font,
            fill=fill,
            spacing=spacing,
            align=align,
            stroke_width=stroke_width,
            stroke_fill=stroke_fill,
        )
        return
    x, y = xy
    lines = text.split("\n") or [text]
    max_w = max(_text_width(draw, line, font, tracking) for line in lines)
    yy = y
    for line in lines:
        line_w = _text_width(draw, line, font, tracking)
        if align == "right":
            xx = x + max_w - line_w
        elif align == "center":
            xx = x + (max_w - line_w) // 2
        else:
            xx = x
        for ch in line:
            draw.text((xx, yy), ch, font=font, fill=fill, stroke_width=stroke_width, stroke_fill=stroke_fill)
            xx += _text_width(draw, ch, font, 0) + tracking
        box = draw.textbbox((0, 0), line or " ", font=font, stroke_width=stroke_width)
        yy += box[3] - box[1] + spacing


def _fit_font(
    draw: ImageDraw.ImageDraw,
    text: str,
    box: tuple[int, int, int, int],
    font_path: Path,
    role: str,
    style: TypographyStyle,
) -> tuple[ImageFont.FreeTypeFont, str]:
    _, _, w, h = box
    scale = style.title_scale if role == "title" else style.body_scale
    start = int(h * scale)
    start = min(start, 164 if role == "title" else 84)
    for size in range(max(18, start), 13, -2):
        font = ImageFont.truetype(str(font_path), size=size)
        tracking = max(0, int(style.tracking * size / 48))
        wrapped = _wrap_text_tracked(draw, text, font, int(w * 0.94), tracking)
        bbox = _tracked_multiline_bbox(draw, wrapped, font, max(4, size // 8), tracking)
        if bbox[2] - bbox[0] <= w * 0.96 and bbox[3] - bbox[1] <= h * 0.92:
            return font, wrapped
    font = ImageFont.truetype(str(font_path), size=14)
    return font, _wrap_text_tracked(draw, text, font, int(w * 0.94), max(0, style.tracking // 2))


def _draw_text_backing(
    base: Image.Image,
    text_rect: tuple[int, int, int, int],
    block: TextBlock,
    style: TypographyStyle,
    font_size: int,
) -> None:
    if style.backing == "none" or style.band_fill[3] <= 0:
        return
    x0, y0, x1, y1 = text_rect
    pad_x = max(10, font_size // 5)
    pad_y = max(6, font_size // 8)
    bx0 = max(block.bbox[0], x0 - pad_x)
    by0 = max(block.bbox[1], y0 - pad_y)
    bx1 = min(block.bbox[0] + block.bbox[2], x1 + pad_x)
    by1 = min(block.bbox[1] + block.bbox[3], y1 + pad_y)
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    if style.backing == "underline":
        line_y = min(block.bbox[1] + block.bbox[3] - 5, y1 + pad_y)
        draw.line((bx0, line_y, bx1, line_y), fill=style.band_fill, width=max(3, font_size // 12))
    elif style.backing == "side_rule":
        draw.rounded_rectangle((bx0, by0, min(bx0 + max(4, font_size // 10), bx1), by1), radius=2, fill=style.band_fill)
    else:
        radius = style.band_radius if style.backing == "tight_panel" else max(style.band_radius, 10)
        draw.rounded_rectangle((bx0, by0, bx1, by1), radius=radius, fill=style.band_fill)
        overlay = overlay.filter(ImageFilter.GaussianBlur(0.25 if style.backing == "tight_panel" else 0.8))
    base.alpha_composite(overlay)


def render_text_layer(
    background_bgr: np.ndarray,
    texts: Sequence[str],
    category: str,
    out_path: Path,
    layout_path: Path | None = None,
    font_path: Path | None = None,
    saliency_aware: bool = True,
    contrast_aware: bool = True,
    style_name: str | None = None,
    layout_prior_path: Path | None = None,
) -> list[TextBlock]:
    style = get_style(style_name, category)
    layout_priors = None
    if layout_prior_path and layout_prior_path.exists():
        priors = json.loads(layout_prior_path.read_text(encoding="utf-8"))
        layout_priors = priors.get(style.name) or priors.get(style.layout_template)
    blocks = plan_layout(
        background_bgr,
        texts,
        category,
        saliency_aware=saliency_aware,
        contrast_aware=contrast_aware,
        layout_template=style.layout_template,
        layout_priors=layout_priors,
    )
    for block in blocks:
        object.__setattr__(block, "color", style.title_color if block.role == "title" else style.body_color)
        object.__setattr__(block, "stroke_color", style.stroke_color)
        object.__setattr__(block, "shadow_color", style.shadow_color)
        object.__setattr__(block, "stroke_width", style.stroke_title if block.role == "title" else style.stroke_body)
    rgb = cv2.cvtColor(background_bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb).convert("RGBA")
    draw = ImageDraw.Draw(image)
    explicit_font = str(font_path) if font_path else None

    draw = ImageDraw.Draw(image)
    render_blocks = []
    for block in blocks:
        x, y, w, h = block.bbox
        text_in = block.text.upper() if style.uppercase else block.text
        role_fonts = style.title_fonts if block.role == "title" else style.body_fonts
        role_font_path = find_font(explicit_font, role_fonts)
        font, text = _fit_font(draw, text_in, block.bbox, role_font_path, block.role, style)
        spacing = max(4, font.size // 8)
        tracking = max(0, int(style.tracking * font.size / 48))
        align = style.align if style.align in {"left", "center", "right"} else "center"
        bbox = _tracked_multiline_bbox(draw, text, font, spacing, tracking, stroke_width=block.stroke_width)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        if align == "left":
            tx = x + max(8, w // 18)
        elif align == "right":
            tx = x + w - tw - max(8, w // 18)
        else:
            tx = x + (w - tw) // 2
        ty = y + (h - th) // 2
        text_rect = (tx + bbox[0], ty + bbox[1], tx + bbox[2], ty + bbox[3])
        _draw_text_backing(image, text_rect, block, style, font.size)
        draw = ImageDraw.Draw(image)
        shadow_offset = max(2, font.size // 18)
        if style.glow:
            for radius, alpha in [(5, 70), (2, 110)]:
                glow = Image.new("RGBA", image.size, (0, 0, 0, 0))
                glow_draw = ImageDraw.Draw(glow)
                _draw_tracked_multiline(
                    glow_draw,
                    (tx, ty),
                    text,
                    font=font,
                    fill=tuple(block.shadow_color) + (alpha,),
                    spacing=spacing,
                    align=align,
                    tracking=tracking,
                )
                image.alpha_composite(glow.filter(ImageFilter.GaussianBlur(radius)))
        else:
            _draw_tracked_multiline(
                draw,
                (tx + shadow_offset, ty + shadow_offset),
                text,
                font=font,
                fill=tuple(block.shadow_color) + (style.shadow_alpha,),
                spacing=spacing,
                align=align,
                tracking=tracking,
            )
        _draw_tracked_multiline(
            draw,
            (tx, ty),
            text,
            font=font,
            fill=tuple(block.color) + (255,),
            stroke_width=block.stroke_width,
            stroke_fill=tuple(block.stroke_color) + (230,),
            spacing=spacing,
            align=align,
            tracking=tracking,
        )
        render_blocks.append(
            {
                "role": block.role,
                "source_text": block.text,
                "rendered_text": text,
                "font_path": str(role_font_path),
                "font_size": font.size,
                "tracking_px": tracking,
                "text_rect": [int(v) for v in text_rect],
                "align": align,
            }
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(out_path, quality=95)
    if layout_path is not None:
        save_layout(layout_path, blocks, image.width, image.height, source=str(out_path))
        payload = json.loads(layout_path.read_text(encoding="utf-8"))
        payload["style"] = style.name
        payload["style_details"] = {
            "backing": style.backing,
            "align": style.align,
            "layout_template": style.layout_template,
            "tracking": style.tracking,
        }
        payload["rendered_blocks"] = render_blocks
        layout_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return blocks
