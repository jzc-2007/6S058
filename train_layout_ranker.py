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

from src.layout import _candidate_boxes, _luminance, _preferred_y, _region, _score_box, edge_density


ROLE_TO_ID = {"title": 0, "subtitle": 1, "meta": 2}


class Ranker(nn.Module):
    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", type=Path, default=Path("prompts/sample_prompts.json"))
    parser.add_argument("--background-dir", type=Path, default=Path("outputs/backgrounds"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/layout_ranker.pt"))
    parser.add_argument("--summary", type=Path, default=Path("results/training_summary.json"))
    parser.add_argument("--curve", type=Path, default=Path("figures/training_curve.png"))
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def features_for_box(image_bgr: np.ndarray, edge: np.ndarray, box: Tuple[int, int, int, int], role: str, preferred_y: float) -> List[float]:
    h_img, w_img = image_bgr.shape[:2]
    x, y, w, h = box
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    crop_edge = _region(edge, box)
    crop_rgb = _region(rgb, box)
    lum = _luminance(crop_rgb) if crop_rgb.size else np.zeros((1, 1), dtype=np.float32)
    margin = min(x, y, w_img - x - w, h_img - y - h) / max(w_img, h_img)
    role_id = ROLE_TO_ID.get(role, 2)
    onehot = [1.0 if role_id == i else 0.0 for i in range(3)]
    center_y = (y + h * 0.5) / h_img
    return [
        x / w_img,
        y / h_img,
        w / w_img,
        h / h_img,
        (w * h) / (w_img * h_img),
        float(np.mean(crop_edge)) if crop_edge.size else 1.0,
        float(np.std(crop_edge)) if crop_edge.size else 0.0,
        float(np.mean(lum) / 255.0),
        float(np.std(lum) / 128.0),
        float(margin),
        abs(center_y - preferred_y),
        *onehot,
    ]


def build_dataset(items: List[Dict], background_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    xs: List[List[float]] = []
    ys: List[float] = []
    for item in items:
        sample_id = item["id"]
        image = cv2.imread(str(background_dir / f"{sample_id}.png"), cv2.IMREAD_COLOR)
        if image is None:
            continue
        height, width = image.shape[:2]
        edge = edge_density(image)
        roles = ["title", "subtitle", "meta"][: len(item["texts"])]
        used: List[Tuple[int, int, int, int]] = []
        for idx, (role, text) in enumerate(zip(roles, item["texts"])):
            candidates = _candidate_boxes(width, height, role, idx, text=text)
            preferred_y = _preferred_y("balanced", role, idx)
            raw_scores = np.array([_score_box(edge, box, used, width, height, preferred_y, role) for box in candidates], dtype=np.float32)
            finite = raw_scores[np.isfinite(raw_scores)]
            mean = float(finite.mean()) if finite.size else 0.0
            std = float(finite.std()) if finite.size and finite.std() > 1e-6 else 1.0
            normalized = (raw_scores - mean) / std
            for box, score in zip(candidates, normalized):
                xs.append(features_for_box(image, edge, box, role, preferred_y))
                ys.append(float(score))
            best_idx = int(np.argmax(raw_scores))
            used.append(candidates[best_idx])
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    items = json.loads(args.prompts.read_text(encoding="utf-8"))
    x, y = build_dataset(items, args.background_dir)
    if len(x) < 20:
        raise RuntimeError("Not enough training examples. Generate backgrounds first.")

    order = np.random.permutation(len(x))
    split = int(len(order) * 0.8)
    train_idx, val_idx = order[:split], order[split:]
    train_ds = TensorDataset(torch.from_numpy(x[train_idx]), torch.from_numpy(y[train_idx]))
    val_x, val_y = torch.from_numpy(x[val_idx]), torch.from_numpy(y[val_idx])
    loader = DataLoader(train_ds, batch_size=64, shuffle=True)

    model = Ranker(x.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    history = []
    for epoch in range(args.epochs):
        model.train()
        train_losses = []
        for bx, by in loader:
            pred = model(bx)
            loss = loss_fn(pred, by)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_losses.append(float(loss.item()))
        model.eval()
        with torch.no_grad():
            val_pred = model(val_x)
            val_loss = float(loss_fn(val_pred, val_y).item())
            corr = float(np.corrcoef(val_pred.numpy(), val_y.numpy())[0, 1])
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(train_losses)), "val_loss": val_loss, "val_corr": corr})

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "in_dim": x.shape[1], "history": history}, args.checkpoint)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "num_examples": int(len(x)),
        "num_train": int(len(train_idx)),
        "num_val": int(len(val_idx)),
        "feature_dim": int(x.shape[1]),
        "epochs": args.epochs,
        "final_train_loss": history[-1]["train_loss"],
        "final_val_loss": history[-1]["val_loss"],
        "final_val_corr": history[-1]["val_corr"],
        "checkpoint": str(args.checkpoint),
    }
    args.summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    args.curve.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.plot([h["train_loss"] for h in history], label="train")
    plt.plot([h["val_loss"] for h in history], label="val")
    plt.xlabel("epoch")
    plt.ylabel("MSE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.curve, dpi=180)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
