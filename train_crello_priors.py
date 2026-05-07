from __future__ import annotations

import argparse
import json
import math
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.styles import STYLE_NAMES


ROLE_TO_ID = {"title": 0, "subtitle": 1, "meta": 2}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="cyberagent/crello")
    parser.add_argument("--split", default="train")
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--metadata-pkl", type=Path, default=None)
    parser.add_argument("--prompts", type=Path, default=Path("prompts/sample_prompts.json"))
    parser.add_argument("--max-records", type=int, default=12000)
    parser.add_argument("--negatives-per-positive", type=int, default=2)
    parser.add_argument("--layout-epochs", type=int, default=10)
    parser.add_argument("--style-epochs", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--layout-checkpoint", type=Path, default=Path("checkpoints/crello_layout_prior.pt"))
    parser.add_argument("--style-checkpoint", type=Path, default=Path("checkpoints/crello_style_lora_adapter.pt"))
    parser.add_argument("--layout-priors", type=Path, default=Path("results/crello_layout_priors.json"))
    parser.add_argument("--style-map", type=Path, default=Path("results/crello_style_predictions.json"))
    parser.add_argument("--summary", type=Path, default=Path("results/crello_training_summary.json"))
    parser.add_argument("--curve", type=Path, default=Path("figures/crello_training_curve.png"))
    return parser.parse_args()


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


class LayoutPriorNet(nn.Module):
    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class LoRALinear(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, rank: int, alpha: float = 8.0) -> None:
        super().__init__()
        self.base = nn.Linear(in_dim, out_dim)
        for p in self.base.parameters():
            p.requires_grad = False
        self.a = nn.Linear(in_dim, rank, bias=False)
        self.b = nn.Linear(rank, out_dim, bias=False)
        nn.init.kaiming_uniform_(self.a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.b.weight)
        self.scale = alpha / rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.b(self.a(x)) * self.scale


class CrelloStyleAdapter(nn.Module):
    def __init__(self, in_dim: int, num_styles: int, rank: int) -> None:
        super().__init__()
        self.l1 = LoRALinear(in_dim, 256, rank)
        self.l2 = LoRALinear(256, 128, rank)
        self.head = nn.Linear(128, num_styles)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        return self.head(x)


def _as_list(row: Dict[str, Any], key: str) -> List[Any]:
    value = row.get(key, [])
    if value is None:
        return []
    return list(value)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _norm_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _layers(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    if isinstance(row.get("layers"), (list, dict)):
        return _layers_from_genposter(row)

    texts = _as_list(row, "text")
    lefts = _as_list(row, "left")
    tops = _as_list(row, "top")
    widths = _as_list(row, "width")
    heights = _as_list(row, "height")
    font_sizes = _as_list(row, "font_size")
    aligns = _as_list(row, "text_align")
    bolds = _as_list(row, "font_bold")
    fonts = _as_list(row, "font")
    canvas_w = max(1.0, _safe_float(row.get("canvas_width"), 1.0))
    canvas_h = max(1.0, _safe_float(row.get("canvas_height"), 1.0))

    out: List[Dict[str, Any]] = []
    n = min(len(texts), len(lefts), len(tops), len(widths), len(heights))
    for i in range(n):
        text = _norm_text(texts[i])
        w, h = _safe_float(widths[i]), _safe_float(heights[i])
        x, y = _safe_float(lefts[i]), _safe_float(tops[i])
        if not text or w <= 4 or h <= 4:
            continue
        if x + w < 0 or y + h < 0 or x > canvas_w or y > canvas_h:
            continue
        fs = _safe_float(font_sizes[i] if i < len(font_sizes) else 0)
        bold_list = bolds[i] if i < len(bolds) and isinstance(bolds[i], list) else []
        out.append(
            {
                "text": text,
                "x": max(0.0, min(x, canvas_w - 1)),
                "y": max(0.0, min(y, canvas_h - 1)),
                "w": max(1.0, min(w, canvas_w)),
                "h": max(1.0, min(h, canvas_h)),
                "font_size": fs,
                "align": int(_safe_float(aligns[i] if i < len(aligns) else 1, 1)),
                "bold": float(np.mean(bold_list)) if bold_list else 0.0,
                "font": str(fonts[i] if i < len(fonts) else ""),
                "canvas_w": canvas_w,
                "canvas_h": canvas_h,
            }
        )
    out.sort(key=lambda z: (z["font_size"] * z["w"] * z["h"], z["w"] * z["h"]), reverse=True)
    for idx, layer in enumerate(out):
        layer["role"] = "title" if idx == 0 else "subtitle" if idx == 1 else "meta"
    return out


def _layers_from_genposter(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw_layers = row.get("layers") or []
    if isinstance(raw_layers, dict):
        keys = list(raw_layers.keys())
        n = max((len(v) for v in raw_layers.values() if isinstance(v, list)), default=0)
        raw_layers = [{key: raw_layers.get(key, [None] * n)[i] for key in keys} for i in range(n)]
    out: List[Dict[str, Any]] = []
    canvas_w = _safe_float(row.get("canvas_width"), 0.0)
    canvas_h = _safe_float(row.get("canvas_height"), 0.0)

    for raw in raw_layers:
        if not isinstance(raw, dict):
            continue
        text = _norm_text(raw.get("text", raw.get("Text")))
        bbox = raw.get("bbox", raw.get("Bounding Box")) or []
        psd_size = raw.get("psd_size", raw.get("PsdSize", raw.get("canvas_size"))) or []
        if len(psd_size) >= 2:
            canvas_w = max(canvas_w, _safe_float(psd_size[0]))
            canvas_h = max(canvas_h, _safe_float(psd_size[1]))
        if len(bbox) != 4 or not text:
            continue
        x0, y0, x1, y1 = [_safe_float(v) for v in bbox]
        x, y = min(x0, x1), min(y0, y1)
        w, h = abs(x1 - x0), abs(y1 - y0)
        canvas_w = max(canvas_w, x + w, 1.0)
        canvas_h = max(canvas_h, y + h, 1.0)
        if w <= 4 or h <= 4:
            continue
        font_size = _safe_float(raw.get("font_size", raw.get("FontSize")), 0.0)
        if font_size <= 0:
            font_size = min(h * 0.75, 160.0)
        align = raw.get("align", raw.get("text_align", raw.get("justification", raw.get("Justification", 1))))
        if isinstance(align, str):
            align = {"left": 0, "center": 1, "right": 2}.get(align.lower(), 1)
        font = raw.get("font", raw.get("Font", ""))
        if isinstance(font, list):
            font = " ".join(map(str, font[:2]))
        out.append(
            {
                "text": text,
                "x": max(0.0, x),
                "y": max(0.0, y),
                "w": max(1.0, w),
                "h": max(1.0, h),
                "font_size": font_size,
                "align": int(_safe_float(align, 1)),
                "bold": 1.0 if "bold" in str(font).lower() else 0.0,
                "font": str(font),
                "canvas_w": max(1.0, canvas_w),
                "canvas_h": max(1.0, canvas_h),
            }
        )

    out.sort(key=lambda z: (z["font_size"] * z["w"] * z["h"], z["w"] * z["h"]), reverse=True)
    for idx, layer in enumerate(out):
        layer["role"] = "title" if idx == 0 else "subtitle" if idx == 1 else "meta"
    return out


def layer_features(layer: Dict[str, Any], role: str | None = None) -> List[float]:
    cw, ch = layer["canvas_w"], layer["canvas_h"]
    role = role or layer["role"]
    role_id = ROLE_TO_ID.get(role, 2)
    nx, ny, nw, nh = layer["x"] / cw, layer["y"] / ch, layer["w"] / cw, layer["h"] / ch
    onehot = [1.0 if role_id == i else 0.0 for i in range(3)]
    return [
        nx,
        ny,
        nw,
        nh,
        nw * nh,
        nw / max(nh, 1e-4),
        nx + nw * 0.5,
        ny + nh * 0.5,
        cw / ch,
        min(len(layer.get("text", "")) / 48.0, 2.0),
        min(layer.get("font_size", 0.0) / max(ch, 1.0) * 12.0, 2.0),
        float(layer.get("align", 1)) / 3.0,
        float(layer.get("bold", 0.0)),
        *onehot,
    ]


def style_features(row: Dict[str, Any], layers: List[Dict[str, Any]]) -> List[float]:
    cw = max(1.0, _safe_float(row.get("canvas_width"), 1.0))
    ch = max(1.0, _safe_float(row.get("canvas_height"), 1.0))
    fs = np.array([x["font_size"] for x in layers], dtype=np.float32) if layers else np.zeros(1, dtype=np.float32)
    ys = np.array([(x["y"] + 0.5 * x["h"]) / x["canvas_h"] for x in layers], dtype=np.float32) if layers else np.zeros(1, dtype=np.float32)
    aligns = [int(x["align"]) for x in layers]
    align_hist = [aligns.count(i) / max(1, len(aligns)) for i in range(4)]
    text_blob = " ".join(
        [
            str(row.get("title", "")),
            " ".join(map(str, row.get("keywords", []) or [])),
            " ".join(x["text"] for x in layers[:4]),
        ]
    ).lower()
    keywords = [
        "sale",
        "music",
        "party",
        "food",
        "business",
        "fashion",
        "beauty",
        "travel",
        "sport",
        "book",
        "art",
        "kids",
    ]
    return [
        cw / ch,
        len(layers) / 8.0,
        float(fs.mean() / max(ch, 1.0) * 12.0),
        float(fs.std() / max(ch, 1.0) * 12.0),
        float(ys.mean()),
        float(ys.std()),
        float(np.mean([x["bold"] for x in layers])) if layers else 0.0,
        *align_hist,
        *[1.0 if k in text_blob else 0.0 for k in keywords],
    ]


def assign_style(row: Dict[str, Any], layers: List[Dict[str, Any]]) -> str:
    text = " ".join(
        [
            str(row.get("title", "")),
            " ".join(map(str, row.get("keywords", []) or [])),
            " ".join(map(str, row.get("industries", []) or [])),
            " ".join(x["text"] for x in layers[:4]),
        ]
    ).lower()
    avg_y = np.mean([(x["y"] + 0.5 * x["h"]) / x["canvas_h"] for x in layers]) if layers else 0.5
    left_rate = np.mean([x["align"] == 0 for x in layers]) if layers else 0.0
    if any(k in text for k in ["music", "party", "club", "festival", "night"]):
        return "retro_pop" if "night" in text and hash(text) % 2 == 0 else "neon"
    if any(k in text for k in ["sport", "fitness", "run", "gym"]):
        return "sport"
    if any(k in text for k in ["food", "restaurant", "market", "organic", "coffee"]):
        return "organic_market"
    if any(k in text for k in ["fashion", "beauty", "cosmetic", "luxury", "sale"]):
        return "luxury_fashion" if any(k in text for k in ["fashion", "beauty", "luxury"]) else "premium_ad"
    if any(k in text for k in ["book", "quote", "poem", "author"]):
        return "editorial_serif"
    if any(k in text for k in ["business", "conference", "webinar", "startup"]):
        return "clean_corporate"
    if any(k in text for k in ["architecture", "lecture", "design school"]):
        return "brutalist_type"
    if any(k in text for k in ["art", "museum", "gallery", "abstract"]):
        return "bauhaus"
    if any(k in text for k in ["wedding", "invitation", "ceremony", "tea"]):
        return "script_invite"
    if any(k in text for k in ["travel", "summer", "beach", "mountain"]):
        return "classic_travel"
    if left_rate > 0.45:
        return "modern_minimal"
    if avg_y > 0.58:
        return "cinematic"
    return "playful_social" if hash(text) % 3 == 0 else "modern_minimal"


def prompt_style(item: Dict[str, Any], idx: int) -> str:
    text = f"{item.get('category','')} {item.get('background_prompt','')} {' '.join(item.get('texts', []))}".lower()
    words = set(re.findall(r"[a-z0-9]+", text))
    if "architecture" in text or "lecture" in text:
        return "brutalist_type"
    if "album" in text or "static" in text:
        return "handmade_marker"
    if "bauhaus" in text or "museum" in text:
        return "bauhaus"
    if "neon" in text or "jazz" in text or "concert" in text or "album" in text:
        return "neon"
    if "cookbook" in text:
        return "script_invite"
    if "book" in text:
        return "editorial_serif"
    if "fitness" in text or "run" in text:
        return "sport"
    if "serum" in text or "fashion" in text:
        return "luxury_fashion"
    if "coffee" in text:
        return "premium_ad"
    if "market" in text or "ramen" in text:
        return "organic_market"
    if "tea" in words:
        return "script_invite"
    if "tech" in text or "summit" in text or "workshop" in text:
        return "space_age" if "summit" in text else "clean_corporate" if idx % 2 else "tech"
    if "travel" in text or "iceland" in text or "bike" in text:
        return "classic_travel" if idx % 2 == 0 else "modern_minimal"
    if "movie" in text:
        return "cinematic"
    if "game" in text or "boardgame" in text:
        return "retro_pop"
    return STYLE_NAMES[idx % len(STYLE_NAMES)]


def _role_prior_mask(arr: np.ndarray, role: str) -> np.ndarray:
    x, y, w, h = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
    aspect = w / np.maximum(h, 1e-4)
    base = (x >= 0.0) & (y >= 0.0) & (x + w <= 1.02) & (y + h <= 1.02)
    if role == "title":
        role_mask = (w >= 0.16) & (w <= 0.94) & (h >= 0.045) & (h <= 0.34) & (aspect >= 1.2)
    elif role == "subtitle":
        role_mask = (w >= 0.12) & (w <= 0.90) & (h >= 0.026) & (h <= 0.18) & (aspect >= 1.7)
    else:
        role_mask = (w >= 0.08) & (w <= 0.74) & (h >= 0.018) & (h <= 0.14) & (aspect >= 1.1)
    return base & role_mask


def _cluster_prior_reps(arr: np.ndarray, role: str, max_reps: int = 10) -> List[List[float]]:
    arr = arr[_role_prior_mask(arr, role)]
    if len(arr) == 0:
        return []
    arr = arr[np.argsort(arr[:, 1] + 0.1 * arr[:, 0])]
    if len(arr) <= max_reps:
        return arr.clip(0.0, 1.0).round(4).tolist()

    k = min(max_reps, max(4, int(np.sqrt(len(arr)) // 9)))
    k = min(k, len(arr))
    scaled = arr * np.array([1.0, 1.0, 0.45, 0.45], dtype=np.float32)
    init_idx = [int(q * (len(arr) - 1)) for q in np.linspace(0.06, 0.94, k)]
    centers = scaled[init_idx].copy()
    for _ in range(18):
        dist = np.sum((scaled[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        labels = np.argmin(dist, axis=1)
        next_centers = centers.copy()
        for j in range(k):
            members = scaled[labels == j]
            if len(members):
                next_centers[j] = members.mean(axis=0)
        if np.max(np.abs(next_centers - centers)) < 1e-5:
            break
        centers = next_centers

    reps = []
    for j in range(k):
        idx = np.where(labels == j)[0]
        if len(idx) == 0:
            continue
        local = scaled[idx]
        medoid = idx[int(np.argmin(np.sum((local - centers[j]) ** 2, axis=1)))]
        reps.append(arr[medoid])
    reps_arr = np.asarray(reps, dtype=np.float32)
    reps_arr = reps_arr[np.argsort(reps_arr[:, 1] + 0.1 * reps_arr[:, 0])]
    return reps_arr.clip(0.0, 1.0).round(4).tolist()


def build_datasets(rows: List[Dict[str, Any]], neg_per_pos: int, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    rng = np.random.default_rng(seed)
    layout_x: List[List[float]] = []
    layout_y: List[float] = []
    style_x: List[List[float]] = []
    style_y: List[int] = []
    box_by_style: Dict[str, Dict[str, List[List[float]]]] = defaultdict(lambda: defaultdict(list))

    for row in rows:
        layers = _layers(row)
        if not layers:
            continue
        style = assign_style(row, layers)
        style_x.append(style_features(row, layers))
        style_y.append(STYLE_NAMES.index(style) if style in STYLE_NAMES else 0)
        for layer in layers:
            role = layer["role"]
            layout_x.append(layer_features(layer, role))
            layout_y.append(1.0)
            box_by_style[style][role].append([layer["x"] / layer["canvas_w"], layer["y"] / layer["canvas_h"], layer["w"] / layer["canvas_w"], layer["h"] / layer["canvas_h"]])
            for _ in range(neg_per_pos):
                neg = dict(layer)
                neg["x"] = float(rng.uniform(0.02, 0.86) * layer["canvas_w"])
                neg["y"] = float(rng.uniform(0.02, 0.88) * layer["canvas_h"])
                neg["w"] = float(rng.uniform(0.18, 0.82) * layer["canvas_w"])
                neg["h"] = float(rng.uniform(0.045, 0.20) * layer["canvas_h"])
                layout_x.append(layer_features(neg, role))
                layout_y.append(0.0)

    priors: Dict[str, Dict[str, List[List[float]]]] = {}
    for style, by_role in box_by_style.items():
        priors[style] = {}
        for role, boxes in by_role.items():
            arr = np.asarray(boxes, dtype=np.float32)
            if len(arr) == 0:
                continue
            reps = _cluster_prior_reps(arr, role, max_reps=10)
            priors[style][role] = reps
    stats = {"num_designs": len(rows), "num_text_designs": len(style_x), "num_layout_examples": len(layout_x)}
    return (
        np.asarray(layout_x, dtype=np.float32),
        np.asarray(layout_y, dtype=np.float32),
        np.asarray(style_x, dtype=np.float32),
        np.asarray(style_y, dtype=np.int64),
        {"priors": priors, "stats": stats},
    )


def train_layout(x: np.ndarray, y: np.ndarray, args: argparse.Namespace, device: torch.device) -> Tuple[LayoutPriorNet, List[Dict[str, float]]]:
    model = LayoutPriorNet(x.shape[1]).to(device)
    order = np.random.permutation(len(x))
    split = int(len(order) * 0.85)
    tr, va = order[:split], order[split:]
    loader = DataLoader(TensorDataset(torch.from_numpy(x[tr]), torch.from_numpy(y[tr])), batch_size=args.batch_size, shuffle=True)
    val_x, val_y = torch.from_numpy(x[va]).to(device), torch.from_numpy(y[va]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    hist = []
    for epoch in range(args.layout_epochs):
        model.train()
        losses = []
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            loss = loss_fn(model(bx), by)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        model.eval()
        with torch.no_grad():
            logits = model(val_x)
            val_loss = float(loss_fn(logits, val_y).item())
            val_acc = float(((torch.sigmoid(logits) > 0.5) == (val_y > 0.5)).float().mean().item())
        hist.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)), "val_loss": val_loss, "val_acc": val_acc})
        log(f"layout epoch {epoch + 1}/{args.layout_epochs}: train_loss={hist[-1]['train_loss']:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.3f}")
    return model, hist


def train_style(x: np.ndarray, y: np.ndarray, args: argparse.Namespace, device: torch.device) -> Tuple[CrelloStyleAdapter, List[Dict[str, float]]]:
    model = CrelloStyleAdapter(x.shape[1], len(STYLE_NAMES), args.rank).to(device)
    order = np.random.permutation(len(x))
    split = int(len(order) * 0.85)
    tr, va = order[:split], order[split:]
    loader = DataLoader(TensorDataset(torch.from_numpy(x[tr]), torch.from_numpy(y[tr])), batch_size=args.batch_size, shuffle=True)
    val_x, val_y = torch.from_numpy(x[va]).to(device), torch.from_numpy(y[va]).to(device)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-3, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    hist = []
    for epoch in range(args.style_epochs):
        model.train()
        losses = []
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            loss = loss_fn(model(bx), by)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        model.eval()
        with torch.no_grad():
            logits = model(val_x)
            val_loss = float(loss_fn(logits, val_y).item())
            val_acc = float((logits.argmax(dim=-1) == val_y).float().mean().item())
        hist.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)), "val_loss": val_loss, "val_acc": val_acc})
        log(f"style epoch {epoch + 1}/{args.style_epochs}: train_loss={hist[-1]['train_loss']:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.3f}")
    return model, hist


def load_rows_from_metadata_pkl(path: Path, max_records: int) -> List[Dict[str, Any]]:
    import pickle

    log(f"loading metadata pkl: {path}")
    with path.open("rb") as f:
        data = pickle.load(f)
    rows: List[Dict[str, Any]] = []
    for i, record in enumerate(data):
        if max_records > 0 and len(rows) >= max_records:
            break
        if isinstance(record, tuple) and len(record) >= 2:
            rows.append(
                {
                    "id": i,
                    "background_image_relpath": record[0],
                    "layers": record[1],
                    "psd_path": record[2] if len(record) > 2 else "",
                    "regions": record[3] if len(record) > 3 else [],
                }
            )
        elif isinstance(record, dict):
            rows.append(record)
        if len(rows) % 1000 == 0 and rows:
            log(f"loaded {len(rows)} metadata records")
    return rows


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.metadata_pkl:
        rows = load_rows_from_metadata_pkl(args.metadata_pkl, args.max_records)
    else:
        from datasets import load_dataset

        log(f"loading dataset={args.dataset} split={args.split} streaming={args.streaming} max_records={args.max_records}")
        if args.streaming:
            ds = load_dataset(args.dataset, split=args.split, streaming=True)
        else:
            split = f"{args.split}[:{args.max_records}]" if args.max_records > 0 else args.split
            ds = load_dataset(args.dataset, split=split)
        rows = []
        for i, row in enumerate(ds, start=1):
            rows.append(row)
            if i == 1:
                log(f"first row keys: {sorted(row.keys())}")
            if i % 500 == 0:
                log(f"loaded {i} records")
            if len(rows) >= args.max_records:
                break
    log(f"building tensors from {len(rows)} records")
    layout_x, layout_y, style_x, style_y, extras = build_datasets(rows, args.negatives_per_positive, args.seed)
    if len(layout_x) < 1000 or len(style_x) < 100:
        raise RuntimeError("Crello extraction produced too few examples.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"training on {device}; layout_examples={len(layout_x)} style_examples={len(style_x)}")
    layout_model, layout_hist = train_layout(layout_x, layout_y, args, device)
    style_model, style_hist = train_style(style_x, style_y, args, device)

    args.layout_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    args.style_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": layout_model.state_dict(), "in_dim": layout_x.shape[1], "history": layout_hist}, args.layout_checkpoint)
    torch.save(
        {"state_dict": style_model.state_dict(), "in_dim": style_x.shape[1], "rank": args.rank, "style_names": STYLE_NAMES, "history": style_hist},
        args.style_checkpoint,
    )

    args.layout_priors.parent.mkdir(parents=True, exist_ok=True)
    args.layout_priors.write_text(json.dumps(extras["priors"], indent=2), encoding="utf-8")
    prompts = json.loads(args.prompts.read_text(encoding="utf-8"))
    style_map = {item["id"]: prompt_style(item, i) for i, item in enumerate(prompts)}
    args.style_map.write_text(json.dumps(style_map, indent=2), encoding="utf-8")

    args.curve.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4))
    plt.plot([h["val_acc"] for h in layout_hist], label="layout prior val acc")
    plt.plot([h["val_acc"] for h in style_hist], label="style LoRA val acc")
    plt.xlabel("epoch")
    plt.ylabel("validation accuracy")
    plt.ylim(0, 1.02)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.curve, dpi=180)

    summary = {
        **extras["stats"],
        "dataset": str(args.metadata_pkl) if args.metadata_pkl else args.dataset,
        "max_records": args.max_records,
        "device": str(device),
        "layout_feature_dim": int(layout_x.shape[1]),
        "style_feature_dim": int(style_x.shape[1]),
        "layout_epochs": args.layout_epochs,
        "style_epochs": args.style_epochs,
        "layout_final_val_acc": layout_hist[-1]["val_acc"],
        "style_final_val_acc": style_hist[-1]["val_acc"],
        "style_names": STYLE_NAMES,
        "style_map": str(args.style_map),
        "layout_priors": str(args.layout_priors),
        "layout_checkpoint": str(args.layout_checkpoint),
        "style_checkpoint": str(args.style_checkpoint),
    }
    args.summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(json.dumps(style_map, indent=2))


if __name__ == "__main__":
    main()
