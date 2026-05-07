from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.styles import STYLE_NAMES
from train_crello_priors import _layers, prompt_style


KEYWORDS = [
    "music",
    "party",
    "club",
    "festival",
    "night",
    "sport",
    "fitness",
    "run",
    "gym",
    "food",
    "restaurant",
    "market",
    "organic",
    "coffee",
    "fashion",
    "beauty",
    "cosmetic",
    "luxury",
    "sale",
    "book",
    "quote",
    "poem",
    "author",
    "business",
    "conference",
    "webinar",
    "startup",
    "architecture",
    "lecture",
    "design",
    "art",
    "museum",
    "gallery",
    "abstract",
    "wedding",
    "invitation",
    "ceremony",
    "tea",
    "travel",
    "summer",
    "beach",
    "mountain",
    "technology",
    "summit",
    "workshop",
    "science",
    "movie",
    "game",
    "photo",
    "climate",
    "jazz",
    "album",
    "cookbook",
]

CATEGORIES = ["travel", "movie", "ad", "museum", "concert", "book", "event", "fitness"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-pkl", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, default=Path("prompts/sample_prompts.json"))
    parser.add_argument("--max-records", type=int, default=60000)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=768)
    parser.add_argument("--hidden-sizes", type=str, default="96,160")
    parser.add_argument("--seeds", type=str, default="13,23,47")
    parser.add_argument("--output-style-map", type=Path, default=Path("results/tool_router_style_predictions.json"))
    parser.add_argument("--summary", type=Path, default=Path("results/tool_router_summary.json"))
    parser.add_argument("--confusion-csv", type=Path, default=Path("results/tool_router_confusion.csv"))
    parser.add_argument("--confusion-figure", type=Path, default=Path("figures/tool_router_confusion.png"))
    return parser.parse_args()


def stable_hash(text: str) -> int:
    return int(hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()[:10], 16)


def text_blob(row: Dict[str, Any], layers: List[Dict[str, Any]]) -> str:
    return " ".join(
        [
            str(row.get("title", "")),
            " ".join(map(str, row.get("keywords", []) or [])),
            " ".join(map(str, row.get("industries", []) or [])),
            " ".join(x["text"] for x in layers[:4]),
        ]
    ).lower()


def assign_tool_style(row: Dict[str, Any], layers: List[Dict[str, Any]]) -> str:
    text = text_blob(row, layers)
    avg_y = np.mean([(x["y"] + 0.5 * x["h"]) / x["canvas_h"] for x in layers]) if layers else 0.5
    left_rate = np.mean([x["align"] == 0 for x in layers]) if layers else 0.0
    if any(k in text for k in ["music", "party", "club", "festival", "night", "jazz", "album"]):
        return "retro_pop" if ("night" in text or stable_hash(text) % 2 == 0) else "neon"
    if any(k in text for k in ["sport", "fitness", "run", "gym"]):
        return "sport"
    if any(k in text for k in ["food", "restaurant", "market", "organic", "coffee", "cookbook"]):
        return "organic_market"
    if any(k in text for k in ["fashion", "beauty", "cosmetic", "luxury", "sale"]):
        return "luxury_fashion" if any(k in text for k in ["fashion", "beauty", "luxury"]) else "premium_ad"
    if any(k in text for k in ["book", "quote", "poem", "author"]):
        return "editorial_serif"
    if any(k in text for k in ["business", "conference", "webinar", "startup", "workshop"]):
        return "clean_corporate"
    if any(k in text for k in ["technology", "summit", "science"]):
        return "space_age" if "summit" in text else "tech"
    if any(k in text for k in ["architecture", "lecture", "design school"]):
        return "brutalist_type"
    if any(k in text for k in ["art", "museum", "gallery", "abstract"]):
        return "bauhaus"
    if any(k in text for k in ["wedding", "invitation", "ceremony", "tea"]):
        return "script_invite"
    if any(k in text for k in ["travel", "summer", "beach", "mountain"]):
        return "classic_travel"
    if "game" in text:
        return "retro_pop"
    if left_rate > 0.45:
        return "modern_minimal"
    if avg_y > 0.58:
        return "cinematic"
    return "playful_social" if stable_hash(text) % 3 == 0 else "modern_minimal"


def router_features(blob: str, category: str, texts: List[str], layer_count: int = 3) -> List[float]:
    words = set(re.findall(r"[a-z0-9]+", blob.lower()))
    keyword = [1.0 if key in blob or key in words else 0.0 for key in KEYWORDS]
    category_onehot = [1.0 if category == c else 0.0 for c in CATEGORIES]
    lengths = [len(t) for t in texts] or [0]
    return [
        *keyword,
        *category_onehot,
        min(len(blob) / 420.0, 2.0),
        min(layer_count / 8.0, 2.0),
        min(float(np.mean(lengths)) / 42.0, 2.0),
        min(float(np.max(lengths)) / 72.0, 2.0),
        min(sum(lengths) / 160.0, 2.0),
    ]


def row_features(row: Dict[str, Any], layers: List[Dict[str, Any]]) -> List[float]:
    blob = text_blob(row, layers)
    texts = [x["text"] for x in layers[:4]]
    return router_features(blob, "", texts, len(layers))


def prompt_features(item: Dict[str, Any]) -> List[float]:
    blob = " ".join(
        [
            item.get("category", ""),
            item.get("background_prompt", ""),
            item.get("baseline_prompt", ""),
            " ".join(item.get("texts", [])),
        ]
    )
    return router_features(blob.lower(), item.get("category", ""), item.get("texts", []), len(item.get("texts", [])))


def load_dataset(path: Path, max_records: int) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    print(f"[{time.strftime('%H:%M:%S')}] loading {path}", flush=True)
    with path.open("rb") as f:
        data = pickle.load(f)
    xs: List[List[float]] = []
    ys: List[int] = []
    for idx, record in enumerate(data):
        if max_records > 0 and idx >= max_records:
            break
        row = {
            "id": idx,
            "background_image_relpath": record[0],
            "layers": record[1],
            "psd_path": record[2] if len(record) > 2 else "",
            "regions": record[3] if len(record) > 3 else [],
        } if isinstance(record, tuple) else record
        if not isinstance(row, dict):
            continue
        layers = _layers(row)
        if not layers:
            continue
        style = assign_tool_style(row, layers)
        if style not in STYLE_NAMES:
            continue
        xs.append(row_features(row, layers))
        ys.append(STYLE_NAMES.index(style))
        if len(xs) % 10000 == 0:
            print(f"[{time.strftime('%H:%M:%S')}] built {len(xs)} router examples", flush=True)
    counts = Counter(STYLE_NAMES[y] for y in ys)
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.int64), dict(counts)


class ToolRouter(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(0.08),
            nn.Linear(hidden, max(32, hidden // 2)),
            nn.GELU(),
            nn.Linear(max(32, hidden // 2), out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_once(x: np.ndarray, y: np.ndarray, hidden: int, seed: int, epochs: int, batch_size: int, device: torch.device) -> Dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    order = np.random.permutation(len(x))
    split = int(len(order) * 0.85)
    tr, va = order[:split], order[split:]
    train_ds = TensorDataset(torch.from_numpy(x[tr]), torch.from_numpy(y[tr]))
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_x, val_y = torch.from_numpy(x[va]).to(device), torch.from_numpy(y[va]).to(device)
    model = ToolRouter(x.shape[1], hidden, len(STYLE_NAMES)).to(device)
    counts = np.bincount(y[tr], minlength=len(STYLE_NAMES)).astype(np.float32)
    weights = np.where(counts > 0, 1.0 / np.sqrt(np.maximum(counts, 1.0)), 0.0)
    weights = weights / max(weights.mean(), 1e-6)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    history = []
    for epoch in range(epochs):
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
            pred = logits.argmax(dim=-1)
            val_acc = float((pred == val_y).float().mean().item())
            val_loss = float(loss_fn(logits, val_y).item())
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)), "val_loss": val_loss, "val_acc": val_acc})
        if epoch == 0 or (epoch + 1) % max(10, epochs // 4) == 0:
            print(f"[router] hidden={hidden} seed={seed} epoch={epoch + 1}/{epochs} val_acc={val_acc:.3f}", flush=True)
    with torch.no_grad():
        logits = model(val_x)
        pred = logits.argmax(dim=-1).cpu().numpy()
        target = val_y.cpu().numpy()
    majority = float(np.max(np.bincount(target, minlength=len(STYLE_NAMES))) / max(1, len(target)))
    confusion = np.zeros((len(STYLE_NAMES), len(STYLE_NAMES)), dtype=np.int64)
    for t, p in zip(target, pred):
        confusion[int(t), int(p)] += 1
    return {
        "model": model,
        "hidden": hidden,
        "seed": seed,
        "history": history,
        "val_acc": history[-1]["val_acc"],
        "val_loss": history[-1]["val_loss"],
        "majority_baseline": majority,
        "confusion": confusion,
    }


def save_confusion(path_csv: Path, path_png: Path, confusion: np.ndarray) -> None:
    path_csv.parent.mkdir(parents=True, exist_ok=True)
    with path_csv.open("w", encoding="utf-8") as f:
        f.write("target,predicted,count\n")
        for i, target in enumerate(STYLE_NAMES):
            for j, pred in enumerate(STYLE_NAMES):
                if confusion[i, j]:
                    f.write(f"{target},{pred},{int(confusion[i, j])}\n")
    row_sums = confusion.sum(axis=1, keepdims=True)
    norm = np.divide(confusion, np.maximum(row_sums, 1), where=row_sums > 0)
    path_png.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 7))
    plt.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    plt.xticks(range(len(STYLE_NAMES)), STYLE_NAMES, rotation=90, fontsize=6)
    plt.yticks(range(len(STYLE_NAMES)), STYLE_NAMES, fontsize=6)
    plt.xlabel("predicted")
    plt.ylabel("weak target")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(path_png, dpi=220)
    plt.close()


def predict_prompts(model: ToolRouter, prompts_path: Path, device: torch.device) -> Tuple[Dict[str, str], Dict[str, Any]]:
    items = json.loads(prompts_path.read_text(encoding="utf-8"))
    model.eval()
    model_map: Dict[str, str] = {}
    hybrid_map: Dict[str, str] = {}
    records = []
    with torch.no_grad():
        for idx, item in enumerate(items):
            x = torch.tensor([prompt_features(item)], dtype=torch.float32, device=device)
            probs = torch.softmax(model(x), dim=-1)[0].cpu().numpy()
            pred_idx = int(np.argmax(probs))
            model_style = STYLE_NAMES[pred_idx]
            rule_style = prompt_style(item, idx)
            confidence = float(probs[pred_idx])
            final_style = model_style if confidence >= 0.34 else rule_style
            model_map[item["id"]] = model_style
            hybrid_map[item["id"]] = final_style
            records.append(
                {
                    "id": item["id"],
                    "model_style": model_style,
                    "rule_fallback_style": rule_style,
                    "final_style": final_style,
                    "confidence": confidence,
                }
            )
    return hybrid_map, {"model_map": model_map, "records": records}


def main() -> None:
    args = parse_args()
    x, y, counts = load_dataset(args.metadata_pkl, args.max_records)
    if len(x) < 1000:
        raise RuntimeError("not enough examples for router training")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[router] device={device} examples={len(x)} feature_dim={x.shape[1]}", flush=True)
    runs = []
    for hidden in [int(v) for v in args.hidden_sizes.split(",") if v.strip()]:
        for seed in [int(v) for v in args.seeds.split(",") if v.strip()]:
            runs.append(train_once(x, y, hidden, seed, args.epochs, args.batch_size, device))
    best = max(runs, key=lambda r: (r["val_acc"], -r["val_loss"]))
    style_map, prompt_info = predict_prompts(best["model"], args.prompts, device)
    args.output_style_map.parent.mkdir(parents=True, exist_ok=True)
    args.output_style_map.write_text(json.dumps(style_map, indent=2), encoding="utf-8")
    save_confusion(args.confusion_csv, args.confusion_figure, best["confusion"])
    summary = {
        "num_examples": int(len(x)),
        "feature_dim": int(x.shape[1]),
        "device": str(device),
        "epochs": args.epochs,
        "runs": [
            {"hidden": r["hidden"], "seed": r["seed"], "val_acc": r["val_acc"], "val_loss": r["val_loss"], "majority_baseline": r["majority_baseline"]}
            for r in runs
        ],
        "best_hidden": best["hidden"],
        "best_seed": best["seed"],
        "best_val_acc": best["val_acc"],
        "majority_baseline": best["majority_baseline"],
        "weak_label_counts": counts,
        "style_map": str(args.output_style_map),
        "confusion_csv": str(args.confusion_csv),
        "confusion_figure": str(args.confusion_figure),
        **prompt_info,
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
