import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import cv2
import numpy as np

try:
    from skimage.metrics import structural_similarity as ssim

    HAS_SSIM = True
except Exception:
    HAS_SSIM = False

from src.layout import layout_validity, load_layout

try:
    import pytesseract

    HAS_OCR = True
except Exception:
    HAS_OCR = False

try:
    import torch
    import lpips

    HAS_LPIPS = True
except Exception:
    HAS_LPIPS = False

_LPIPS_MODEL = None
_LPIPS_DEVICE = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", type=Path, default=Path("prompts/sample_prompts.json"))
    parser.add_argument("--ours-dir", type=Path, default=Path("outputs/ours"))
    parser.add_argument("--baseline-dir", type=Path, default=Path("outputs/baselines"))
    parser.add_argument("--layout-dir", type=Path, default=Path("outputs/layouts"))
    parser.add_argument("--edit-before-dir", type=Path, default=Path("outputs/edit_before"))
    parser.add_argument("--edit-after-dir", type=Path, default=Path("outputs/edit_after"))
    parser.add_argument("--edit-before-layout-dir", type=Path, default=Path("outputs/edit_before_layouts"))
    parser.add_argument("--edit-layout-dir", type=Path, default=Path("outputs/edit_layouts"))
    parser.add_argument("--output-csv", type=Path, default=Path("results/metrics.csv"))
    return parser.parse_args()


def norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _ocr_image(image: np.ndarray, psm: int = 6) -> str:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    texts = []
    for candidate in [
        gray,
        cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
        cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1],
    ]:
        try:
            texts.append(pytesseract.image_to_string(candidate, config=f"--psm {psm}"))
        except Exception:
            pass
    return "\n".join(texts)


def ocr_text(path: Path) -> str:
    if not HAS_OCR or not path.exists():
        return ""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        return ""
    return _ocr_image(image, psm=6)


def exact_match(path: Path, expected: List[str]) -> float:
    observed = norm(ocr_text(path))
    if not observed:
        return float("nan")
    hits = sum(1 for target in expected if norm(target) in observed)
    return hits / max(1, len(expected))


def word_recall_from_text(observed: str, expected: List[str]) -> float:
    obs_words = set(norm(observed).split())
    target_words = []
    for target in expected:
        target_words.extend(norm(target).split())
    if not target_words:
        return float("nan")
    return sum(1 for word in target_words if word in obs_words) / len(target_words)


def _crop_boxes(layout_path: Path) -> List[Tuple[int, int, int, int]]:
    if not layout_path.exists():
        return []
    payload = json.loads(layout_path.read_text(encoding="utf-8"))
    rendered = payload.get("rendered_blocks") or []
    boxes = []
    for idx, block in enumerate(payload.get("blocks", [])):
        if idx < len(rendered) and "text_rect" in rendered[idx]:
            x0, y0, x1, y1 = [int(v) for v in rendered[idx]["text_rect"]]
            boxes.append((x0, y0, max(1, x1 - x0), max(1, y1 - y0)))
        else:
            x, y, w, h = [int(v) for v in block["bbox"]]
            boxes.append((x, y, w, h))
    return boxes


def exact_match_crops(path: Path, layout_path: Path, expected: List[str]) -> float:
    if not HAS_OCR or not path.exists() or not layout_path.exists():
        return float("nan")
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        return float("nan")
    hits = 0
    for (x, y, w, h), target in zip(_crop_boxes(layout_path), expected):
        pad = 10
        crop = image[max(0, y - pad) : min(image.shape[0], y + h + pad), max(0, x - pad) : min(image.shape[1], x + w + pad)]
        observed = norm(_ocr_image(crop, psm=6))
        hits += int(norm(target) in observed)
    return hits / max(1, len(expected))


def word_recall_crops(path: Path, layout_path: Path, expected: List[str]) -> float:
    if not HAS_OCR or not path.exists() or not layout_path.exists():
        return float("nan")
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        return float("nan")
    recalls = []
    for (x, y, w, h), target in zip(_crop_boxes(layout_path), expected):
        pad = 20
        crop = image[max(0, y - pad) : min(image.shape[0], y + h + pad), max(0, x - pad) : min(image.shape[1], x + w + pad)]
        recalls.append(word_recall_from_text(_ocr_image(crop, psm=6), [target]))
    return float(np.mean(recalls)) if recalls else float("nan")


def contrast_score(image_path: Path, layout_path: Path) -> float:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None or not layout_path.exists():
        return float("nan")
    blocks = load_layout(layout_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    vals = []
    for block in blocks:
        x, y, w, h = block.bbox
        crop = gray[y : y + h, x : x + w]
        if crop.size:
            vals.append(float(np.std(crop) / 128.0))
    return float(np.mean(vals)) if vals else float("nan")


def edit_text_mask(shape: Tuple[int, int], layout_paths: Sequence[Path], pad: int = 16) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    for layout_path in layout_paths:
        if not layout_path.exists():
            continue
        for x, y, w, h in _crop_boxes(layout_path):
            x0, y0 = max(0, x - pad), max(0, y - pad)
            x1, y1 = min(shape[1], x + w + pad), min(shape[0], y + h + pad)
            mask[y0:y1, x0:x1] = 1
    return mask


def _edit_images(before_path: Path, after_path: Path) -> Tuple[np.ndarray | None, np.ndarray | None]:
    if not before_path.exists() or not after_path.exists():
        return None, None
    before = cv2.imread(str(before_path), cv2.IMREAD_COLOR)
    after = cv2.imread(str(after_path), cv2.IMREAD_COLOR)
    if before is None or after is None:
        return None, None
    if before.shape != after.shape:
        after = cv2.resize(after, (before.shape[1], before.shape[0]))
    return before, after


def edit_bg_ssim(before_path: Path, after_path: Path, layout_paths: Sequence[Path]) -> float:
    if not HAS_SSIM:
        return float("nan")
    before, after = _edit_images(before_path, after_path)
    if before is None or after is None:
        return float("nan")
    text_mask = edit_text_mask(before.shape[:2], layout_paths)
    bg_mask = text_mask == 0
    before_gray = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
    after_gray = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)
    score, sim_map = ssim(before_gray, after_gray, data_range=255, full=True)
    if np.any(bg_mask):
        return float(np.mean(sim_map[bg_mask]))
    return float(score)


def _lpips_model():
    global HAS_LPIPS, _LPIPS_MODEL, _LPIPS_DEVICE
    if not HAS_LPIPS:
        return None, None
    if _LPIPS_MODEL is None:
        try:
            _LPIPS_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            _LPIPS_MODEL = lpips.LPIPS(net="alex").to(_LPIPS_DEVICE)
            _LPIPS_MODEL.eval()
        except Exception as exc:
            print(f"[warn] LPIPS initialization failed: {exc}")
            HAS_LPIPS = False
            return None, None
    return _LPIPS_MODEL, _LPIPS_DEVICE


def _lpips_tensor(image_bgr: np.ndarray, device) -> Any:
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
    return (tensor / 127.5 - 1.0).to(device)


def edit_bg_lpips(before_path: Path, after_path: Path, layout_paths: Sequence[Path]) -> float:
    model, device = _lpips_model()
    if model is None or device is None:
        return float("nan")
    before, after = _edit_images(before_path, after_path)
    if before is None or after is None:
        return float("nan")
    text_mask = edit_text_mask(before.shape[:2], layout_paths).astype(bool)
    if np.any(text_mask):
        before = before.copy()
        after = after.copy()
        before[text_mask] = 127
        after[text_mask] = 127
    with torch.no_grad():
        value = model(_lpips_tensor(before, device), _lpips_tensor(after, device))
    return float(value.item())


def mean(values: List[float]) -> float:
    vals = [v for v in values if not np.isnan(v)]
    return float(np.mean(vals)) if vals else float("nan")


def json_safe(value: Any) -> Any:
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


def main() -> None:
    args = parse_args()
    items: List[Dict[str, Any]] = json.loads(args.prompts.read_text(encoding="utf-8"))
    rows = []
    for item in items:
        sample_id = item["id"]
        ours = args.ours_dir / f"{sample_id}.png"
        baseline = args.baseline_dir / f"{sample_id}.png"
        layout = args.layout_dir / f"{sample_id}.json"
        edit_layouts = [
            args.edit_before_layout_dir / f"{sample_id}.json",
            args.edit_layout_dir / f"{sample_id}.json",
        ]
        if not ours.exists() or not layout.exists():
            continue
        blocks = load_layout(layout)
        image = cv2.imread(str(ours), cv2.IMREAD_COLOR)
        valid = layout_validity(blocks, image.shape[1], image.shape[0]) if image is not None else float("nan")
        row = {
            "id": sample_id,
            "rendered_text_exact_ours": 1.0,
            "ocr_exact_ours": exact_match_crops(ours, layout, item["texts"]),
            "ocr_exact_baseline": exact_match(baseline, item["texts"]) if baseline.exists() else float("nan"),
            "ocr_word_recall_ours": word_recall_crops(ours, layout, item["texts"]),
            "ocr_word_recall_baseline": word_recall_from_text(ocr_text(baseline), item["texts"]) if baseline.exists() else float("nan"),
            "layout_validity": valid,
            "contrast_proxy": contrast_score(ours, layout),
            "edit_bg_ssim": edit_bg_ssim(
                args.edit_before_dir / f"{sample_id}.png",
                args.edit_after_dir / f"{sample_id}.png",
                edit_layouts,
            ),
            "edit_bg_lpips": edit_bg_lpips(
                args.edit_before_dir / f"{sample_id}.png",
                args.edit_after_dir / f"{sample_id}.png",
                edit_layouts,
            ),
        }
        rows.append(row)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "id",
        "rendered_text_exact_ours",
        "ocr_exact_ours",
        "ocr_exact_baseline",
        "ocr_word_recall_ours",
        "ocr_word_recall_baseline",
        "layout_validity",
        "contrast_proxy",
        "edit_bg_ssim",
        "edit_bg_lpips",
    ]
    with args.output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {key: mean([float(r[key]) for r in rows]) for key in fieldnames if key != "id"}
    if args.output_csv.name == "metrics.csv":
        summary_path = args.output_csv.parent / "summary.json"
    else:
        summary_path = args.output_csv.with_name(args.output_csv.stem.replace("metrics", "summary") + ".json")
    summary_path.write_text(json.dumps({k: json_safe(v) for k, v in summary.items()}, indent=2), encoding="utf-8")
    print(f"[saved] {args.output_csv}")
    print(json.dumps(summary, indent=2))
    if not HAS_OCR:
        print("[warn] pytesseract import failed; OCR metrics are NaN.")
    if not HAS_SSIM:
        print("[warn] skimage import failed; edit SSIM metrics are NaN.")
    if not HAS_LPIPS:
        print("[warn] lpips/torch import failed; edit LPIPS metrics are NaN.")


if __name__ == "__main__":
    main()
