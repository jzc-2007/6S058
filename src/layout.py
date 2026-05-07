from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import cv2
import numpy as np


Box = Tuple[int, int, int, int]
Color = Tuple[int, int, int]


@dataclass
class TextBlock:
    text: str
    role: str
    bbox: Box
    color: Color
    stroke_color: Color
    shadow_color: Color
    stroke_width: int
    align: str
    font_scale: float


def edge_density(image_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(grad_x, grad_y)
    mag = cv2.GaussianBlur(mag, (31, 31), 0)
    denom = float(np.percentile(mag, 99))
    return np.clip(mag / max(denom, 1e-6), 0, 1)


def _region(arr: np.ndarray, box: Box) -> np.ndarray:
    x, y, w, h = box
    return arr[max(0, y) : max(0, y + h), max(0, x) : max(0, x + w)]


def _inside(box: Box, width: int, height: int) -> bool:
    x, y, w, h = box
    return x >= 0 and y >= 0 and x + w <= width and y + h <= height


def _overlap_area(a: Box, b: Box) -> int:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix0, iy0 = max(ax, bx), max(ay, by)
    ix1, iy1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    return max(0, ix1 - ix0) * max(0, iy1 - iy0)


def _luminance(rgb: np.ndarray) -> np.ndarray:
    return 0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]


def _pick_color(image_bgr: np.ndarray, box: Box, category: str, force_contrast: bool = True) -> Tuple[Color, Color, Color]:
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    crop = _region(rgb, box)
    if crop.size == 0:
        return (248, 248, 244), (12, 12, 12), (0, 0, 0)

    mean_lum = float(np.mean(_luminance(crop)))
    accent_by_category = {
        "concert": (255, 236, 91),
        "museum": (240, 50, 45),
        "tech": (130, 220, 255),
        "event": (250, 250, 245),
        "book": (238, 232, 218),
        "travel": (255, 247, 220),
        "movie": (245, 245, 245),
        "ad": (255, 255, 255),
    }
    accent = accent_by_category.get(category, (245, 245, 245))
    if force_contrast:
        return (255, 255, 255), (0, 0, 0), (0, 0, 0)
    if not force_contrast:
        return accent, (15, 15, 15), (0, 0, 0)
    if mean_lum > 150:
        return (18, 18, 18), (250, 250, 250), (255, 255, 255)
    if mean_lum < 70:
        return accent, (4, 4, 4), (0, 0, 0)
    return (250, 250, 245), (8, 8, 8), (0, 0, 0)


def _template_positions(width: int, height: int, role: str, index: int, template: str) -> Tuple[List[float], List[float]]:
    if template == "top_left":
        return [0.06, 0.10, 0.16, 0.48], [0.06, 0.16, 0.30, 0.74, 0.86]
    if template == "left_column":
        return [0.06, 0.10, 0.14, 0.20], [0.08, 0.22, 0.46, 0.70, 0.84]
    if template == "bottom_title":
        if role == "title":
            return [0.06, 0.12, 0.18, 0.50], [0.58, 0.68, 0.76, 0.08]
        return [0.08, 0.16, 0.42, 0.54], [0.08, 0.20, 0.78, 0.86]
    if template == "center_stack":
        return [0.07, 0.12, 0.18, 0.50], [0.12, 0.26, 0.42, 0.60, 0.78]
    if template == "editorial":
        if role == "title":
            return [0.06, 0.10, 0.46, 0.54], [0.06, 0.14, 0.52, 0.68]
        return [0.08, 0.48, 0.56, 0.16], [0.22, 0.62, 0.74, 0.84]
    if template == "dynamic_diagonal":
        return [0.05, 0.14, 0.26, 0.42], [0.07, 0.20, 0.46, 0.66, 0.82]
    return [0.07, 0.12, 0.18, 0.50], [0.06, 0.18, 0.48, 0.67, 0.80]


LayoutPriors = Dict[str, Dict[str, List[List[float]]]]


def _role_box_size(width: int, height: int, role: str, text: str = "") -> Tuple[int, int]:
    chars = max(1, len(text.replace(" ", "")))
    words = max(1, len(text.split()))
    long_title = role == "title" and (chars > 18 or words > 3)
    short_title = role == "title" and chars <= 11
    if role == "title":
        box_w = int(width * (0.88 if not short_title else 0.76))
        box_h = int(height * (0.22 if long_title else 0.17))
    elif role == "subtitle":
        box_w = int(width * (0.80 if chars > 20 else 0.70))
        box_h = int(height * (0.105 if chars > 24 else 0.082))
    else:
        box_w = int(width * (0.62 if chars > 18 else 0.48))
        box_h = int(height * 0.064)
    return box_w, box_h


def _clamp_box(box: Box, width: int, height: int, margin: int) -> Box:
    x, y, w, h = box
    w = min(max(1, w), max(1, width - 2 * margin))
    h = min(max(1, h), max(1, height - 2 * margin))
    x = min(max(margin, x), max(margin, width - margin - w))
    y = min(max(margin, y), max(margin, height - margin - h))
    return (int(x), int(y), int(w), int(h))


def _preferred_y(template: str, role: str, index: int) -> float:
    if template == "bottom_title":
        return {"title": 0.76, "subtitle": 0.18, "meta": 0.88}.get(role, 0.82)
    if template in {"top_left", "left_column"}:
        return {"title": 0.17, "subtitle": 0.34, "meta": 0.74}.get(role, 0.74)
    if template == "editorial":
        return {"title": 0.24 if index == 0 else 0.58, "subtitle": 0.68, "meta": 0.82}.get(role, 0.70)
    if template == "center_stack":
        return {"title": 0.32, "subtitle": 0.58, "meta": 0.78}.get(role, 0.78)
    if template == "dynamic_diagonal":
        return {"title": 0.22, "subtitle": 0.58, "meta": 0.82}.get(role, 0.82)
    return {"title": 0.20, "subtitle": 0.68, "meta": 0.84}.get(role, 0.82)


def _candidate_boxes(
    width: int,
    height: int,
    role: str,
    index: int,
    template: str = "balanced",
    priors: Dict[str, List[List[float]]] | None = None,
    text: str = "",
) -> List[Box]:
    box_w, box_h = _role_box_size(width, height, role, text)
    margin = max(18, int(min(width, height) * 0.035))

    x_fracs, y_fracs = _template_positions(width, height, role, index, template)
    xs = [int(width * x) for x in x_fracs[:-1]] + [int(width * x_fracs[-1] - box_w * 0.5)]
    ys = [int(height * y) for y in y_fracs]
    if template == "balanced":
        if index == 1:
            ys = [int(height * v) for v in [0.22, 0.70, 0.83, 0.10]]
        elif index >= 2:
            ys = [int(height * v) for v in [0.84, 0.74, 0.12, 0.58]]
    if template in {"left_column", "top_left", "editorial"}:
        box_w = int(box_w * (0.84 if role == "title" else 0.72))
    if template == "dynamic_diagonal" and index % 2 == 1:
        xs = [min(width - box_w, x + int(width * 0.18)) for x in xs]

    boxes = []
    for y in ys:
        for x in xs:
            boxes.append(_clamp_box((x, y, box_w, box_h), width, height, margin))

    wide_w, wide_h = int(width * 0.88), int(height * (0.20 if role == "title" else 0.08))
    side_w, side_h = int(width * 0.58), int(height * (0.24 if role == "title" else 0.10))
    anchor_boxes = {
        "title": [
            (int(width * 0.06), int(height * 0.07), wide_w, wide_h),
            (int(width * 0.06), int(height * 0.66), wide_w, wide_h),
            (int(width * 0.20), int(height * 0.36), int(width * 0.60), int(height * 0.22)),
            (int(width * 0.07), int(height * 0.18), side_w, side_h),
        ],
        "subtitle": [
            (int(width * 0.09), int(height * 0.22), int(width * 0.74), int(height * 0.10)),
            (int(width * 0.10), int(height * 0.58), int(width * 0.72), int(height * 0.10)),
            (int(width * 0.12), int(height * 0.78), int(width * 0.68), int(height * 0.08)),
        ],
        "meta": [
            (int(width * 0.10), int(height * 0.84), int(width * 0.58), int(height * 0.065)),
            (int(width * 0.54), int(height * 0.08), int(width * 0.36), int(height * 0.06)),
            (int(width * 0.10), int(height * 0.08), int(width * 0.42), int(height * 0.06)),
        ],
    }
    boxes.extend(_clamp_box(b, width, height, margin) for b in anchor_boxes.get(role, []))
    if priors:
        for prior in priors.get(role, []):
            nx, ny, nw, nh = prior[:4]
            if role == "title":
                min_w, min_h, max_w, max_h = 0.54, 0.13, 0.92, 0.28
            elif role == "subtitle":
                min_w, min_h, max_w, max_h = 0.38, 0.060, 0.86, 0.15
            else:
                min_w, min_h, max_w, max_h = 0.26, 0.045, 0.68, 0.12
            pw = min(max(int(width * nw), int(width * min_w)), int(width * max_w))
            ph = min(max(int(height * nh), int(height * min_h)), int(height * max_h))
            px = int(width * nx)
            py = int(height * ny)
            boxes.append(_clamp_box((px, py, pw, ph), width, height, margin))

    deduped: List[Box] = []
    seen = set()
    for box in boxes:
        key = tuple(round(v / 8) for v in box)
        if key not in seen and _inside(box, width, height):
            seen.add(key)
            deduped.append(box)
    return deduped


def _score_box(edge: np.ndarray, box: Box, used: Sequence[Box], width: int, height: int, preferred_y: float, role: str = "meta") -> float:
    if not _inside(box, width, height):
        return -1e9
    x, y, w, h = box
    complexity = float(np.mean(_region(edge, box)))
    area = max(1, w * h)
    overlap = sum(_overlap_area(box, other) / max(1, min(area, other[2] * other[3])) for other in used)
    close_penalty = 0.0
    for other in used:
        ox, oy, ow, oh = other
        gap_x = max(0, max(ox - (x + w), x - (ox + ow)))
        gap_y = max(0, max(oy - (y + h), y - (oy + oh)))
        if gap_x < width * 0.035 and gap_y < height * 0.035:
            close_penalty += 0.35
    center_y = (y + h * 0.5) / height
    y_penalty = abs(center_y - preferred_y) * 0.26
    edge_margin = min(x, y, width - x - w, height - y - h)
    margin_bonus = min(0.12, edge_margin / max(width, height))
    size_bonus = 0.10 * (w / width) + (0.08 if role == "title" else 0.03) * (h / height)
    return -0.85 * complexity - 10.0 * overlap - close_penalty - y_penalty + margin_bonus + size_bonus


def plan_layout(
    image_bgr: np.ndarray,
    texts: Sequence[str],
    category: str = "poster",
    saliency_aware: bool = True,
    contrast_aware: bool = True,
    layout_template: str = "balanced",
    layout_priors: Dict[str, List[List[float]]] | None = None,
) -> List[TextBlock]:
    height, width = image_bgr.shape[:2]
    edge = edge_density(image_bgr) if saliency_aware else np.zeros((height, width), dtype=np.float32)
    roles = ["title", "subtitle", "meta", "meta"][: len(texts)]
    blocks: List[TextBlock] = []
    used: List[Box] = []

    for i, (text, role) in enumerate(zip(texts, roles)):
        candidates = _candidate_boxes(width, height, role, i, layout_template, layout_priors, text)
        preferred_y = _preferred_y(layout_template, role, i)
        best = max(candidates, key=lambda b: _score_box(edge, b, used, width, height, preferred_y, role))
        color, stroke, shadow = _pick_color(image_bgr, best, category, force_contrast=contrast_aware)
        blocks.append(
            TextBlock(
                text=text,
                role=role,
                bbox=best,
                color=color,
                stroke_color=stroke,
                shadow_color=shadow,
                stroke_width=4 if role == "title" else 2,
                align="center",
                font_scale=1.0 if role == "title" else 0.82,
            )
        )
        used.append(best)
    return blocks


def layout_validity(blocks: Sequence[TextBlock], width: int, height: int) -> float:
    ok = 0
    for i, block in enumerate(blocks):
        inside = _inside(block.bbox, width, height)
        no_overlap = all(_overlap_area(block.bbox, other.bbox) == 0 for j, other in enumerate(blocks) if i != j)
        ok += int(inside and no_overlap)
    return ok / max(1, len(blocks))


def save_layout(path: Path, blocks: Iterable[TextBlock], width: int, height: int, source: str) -> None:
    payload = {"width": width, "height": height, "source": source, "blocks": [asdict(b) for b in blocks]}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_layout(path: Path) -> List[TextBlock]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [TextBlock(**block) for block in payload["blocks"]]
