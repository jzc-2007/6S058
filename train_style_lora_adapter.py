import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.layout import edge_density
from src.styles import STYLE_NAMES, style_for_category


KEYWORDS = [
    "travel",
    "vintage",
    "cinematic",
    "movie",
    "coffee",
    "premium",
    "museum",
    "bauhaus",
    "concert",
    "neon",
    "book",
    "minimal",
    "restaurant",
    "ramen",
    "technology",
    "conference",
    "fitness",
    "education",
    "science",
]

CATEGORIES = ["travel", "movie", "ad", "museum", "concert", "book", "event", "fitness"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", type=Path, default=Path("prompts/sample_prompts.json"))
    parser.add_argument("--background-dir", type=Path, default=Path("outputs/backgrounds"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/style_lora_adapter.pt"))
    parser.add_argument("--style-map", type=Path, default=Path("results/style_predictions.json"))
    parser.add_argument("--summary", type=Path, default=Path("results/style_training_summary.json"))
    parser.add_argument("--curve", type=Path, default=Path("figures/style_training_curve.png"))
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--augment", type=int, default=64)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--seed", type=int, default=11)
    return parser.parse_args()


class LoRALinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, rank: int, alpha: float = 8.0) -> None:
        super().__init__()
        self.base = nn.Linear(in_features, out_features)
        for param in self.base.parameters():
            param.requires_grad = False
        self.a = nn.Linear(in_features, rank, bias=False)
        self.b = nn.Linear(rank, out_features, bias=False)
        nn.init.kaiming_uniform_(self.a.weight, a=np.sqrt(5))
        nn.init.zeros_(self.b.weight)
        self.scale = alpha / rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.b(self.a(x)) * self.scale


class StyleLoRAAdapter(nn.Module):
    def __init__(self, in_dim: int, num_styles: int, rank: int) -> None:
        super().__init__()
        self.lora1 = LoRALinear(in_dim, 96, rank)
        self.lora2 = LoRALinear(96, 48, rank)
        self.head = nn.Linear(48, num_styles)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.lora1(x))
        x = torch.relu(self.lora2(x))
        return self.head(x)


def target_style(item: Dict) -> str:
    category = item.get("category", "")
    prompt = (item.get("background_prompt", "") + " " + item.get("baseline_prompt", "")).lower()
    if "bauhaus" in prompt or category == "museum":
        return "bauhaus"
    if "neon" in prompt or category == "concert":
        return "neon"
    if "book" in prompt or category == "book":
        return "literary"
    if "technology" in prompt or "conference" in prompt:
        return "tech"
    if "fitness" in prompt:
        return "sport"
    if "coffee" in prompt or "restaurant" in prompt or category == "ad":
        return "premium_ad"
    if category == "travel":
        return "classic_travel"
    return style_for_category(category)


def image_features(image_bgr: np.ndarray) -> List[float]:
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    edge = edge_density(image_bgr)
    means = rgb.reshape(-1, 3).mean(axis=0).tolist()
    stds = rgb.reshape(-1, 3).std(axis=0).tolist()
    return [
        float(gray.mean()),
        float(gray.std()),
        float(edge.mean()),
        float(edge.std()),
        *[float(v) for v in means],
        *[float(v) for v in stds],
    ]


def text_features(item: Dict) -> List[float]:
    text = (item.get("category", "") + " " + item.get("background_prompt", "") + " " + item.get("baseline_prompt", "")).lower()
    keyword = [1.0 if word in text else 0.0 for word in KEYWORDS]
    category = [1.0 if item.get("category", "") == c else 0.0 for c in CATEGORIES]
    strings = item.get("texts", [])
    avg_len = np.mean([len(s) for s in strings]) / 32.0 if strings else 0.0
    max_len = max([len(s) for s in strings], default=0) / 48.0
    return keyword + category + [float(avg_len), float(max_len), float(len(strings) / 4.0)]


def make_dataset(items: List[Dict], background_dir: Path, augment: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    xs: List[List[float]] = []
    ys: List[int] = []
    for item in items:
        image = cv2.imread(str(background_dir / f"{item['id']}.png"), cv2.IMREAD_COLOR)
        if image is None:
            continue
        base = np.array(text_features(item) + image_features(image), dtype=np.float32)
        label = STYLE_NAMES.index(target_style(item))
        for _ in range(augment):
            noise = rng.normal(0, 0.035, size=base.shape).astype(np.float32)
            # Do not perturb binary prompt/category features as much as continuous image stats.
            noise[: len(KEYWORDS) + len(CATEGORIES)] *= 0.15
            xs.append(np.clip(base + noise, 0, 2).tolist())
            ys.append(label)
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.int64)


def predict_style_map(model: StyleLoRAAdapter, items: List[Dict], background_dir: Path) -> Dict[str, str]:
    model.eval()
    out: Dict[str, str] = {}
    with torch.no_grad():
        for item in items:
            image = cv2.imread(str(background_dir / f"{item['id']}.png"), cv2.IMREAD_COLOR)
            if image is None:
                out[item["id"]] = target_style(item)
                continue
            x = torch.tensor([text_features(item) + image_features(image)], dtype=torch.float32)
            pred = int(torch.argmax(model(x), dim=-1).item())
            out[item["id"]] = STYLE_NAMES[pred]
    return out


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    items = json.loads(args.prompts.read_text(encoding="utf-8"))
    x, y = make_dataset(items, args.background_dir, args.augment, args.seed)
    if len(x) < 100:
        raise RuntimeError("Not enough style training examples. Generate backgrounds first.")

    order = np.random.permutation(len(x))
    split = int(len(order) * 0.8)
    train_idx, val_idx = order[:split], order[split:]
    train_ds = TensorDataset(torch.from_numpy(x[train_idx]), torch.from_numpy(y[train_idx]))
    val_x, val_y = torch.from_numpy(x[val_idx]), torch.from_numpy(y[val_idx])
    loader = DataLoader(train_ds, batch_size=64, shuffle=True)

    model = StyleLoRAAdapter(x.shape[1], len(STYLE_NAMES), args.rank)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-3, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    history = []
    for epoch in range(args.epochs):
        model.train()
        train_loss = []
        for bx, by in loader:
            logits = model(bx)
            loss = loss_fn(logits, by)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss.append(float(loss.item()))
        model.eval()
        with torch.no_grad():
            val_logits = model(val_x)
            val_loss = float(loss_fn(val_logits, val_y).item())
            val_acc = float((val_logits.argmax(dim=-1) == val_y).float().mean().item())
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(train_loss)), "val_loss": val_loss, "val_acc": val_acc})

    style_map = predict_style_map(model, items, args.background_dir)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "style_names": STYLE_NAMES,
            "keywords": KEYWORDS,
            "categories": CATEGORIES,
            "in_dim": int(x.shape[1]),
            "rank": args.rank,
            "history": history,
            "style_map": style_map,
        },
        args.checkpoint,
    )
    args.style_map.parent.mkdir(parents=True, exist_ok=True)
    args.style_map.write_text(json.dumps(style_map, indent=2), encoding="utf-8")

    summary = {
        "num_examples": int(len(x)),
        "num_train": int(len(train_idx)),
        "num_val": int(len(val_idx)),
        "feature_dim": int(x.shape[1]),
        "num_styles": len(STYLE_NAMES),
        "styles": STYLE_NAMES,
        "rank": args.rank,
        "epochs": args.epochs,
        "final_train_loss": history[-1]["train_loss"],
        "final_val_loss": history[-1]["val_loss"],
        "final_val_acc": history[-1]["val_acc"],
        "checkpoint": str(args.checkpoint),
        "style_map": str(args.style_map),
    }
    args.summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    args.curve.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.plot([h["train_loss"] for h in history], label="train loss")
    plt.plot([h["val_loss"] for h in history], label="val loss")
    plt.xlabel("epoch")
    plt.ylabel("cross entropy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.curve, dpi=180)
    print(json.dumps(summary, indent=2))
    print(json.dumps(style_map, indent=2))


if __name__ == "__main__":
    main()
