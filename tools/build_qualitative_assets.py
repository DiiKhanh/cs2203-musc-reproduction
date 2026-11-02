#!/usr/bin/env python3
"""Build original, ground-truth, heatmap, and overlay images for selected cases.

ground_truth.png keeps raw mask values. VisA masks store class IDs from 1 to 7,
so display them with `mask > 0`, as tools/render_slide_strips.py does.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

IMAGE_SIZE = 518
ORIGINAL_WEIGHT = 0.55
HEATMAP_WEIGHT = 0.45
CONTOUR_PX = 2


def load_original(path: Path) -> np.ndarray:
    """Resize to the square MuSc input, which changes the source aspect ratio."""
    image = Image.open(path).convert("RGB")
    return np.asarray(image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BICUBIC))


def load_heatmap(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Cannot read heatmap: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if rgb.shape[:2] != (IMAGE_SIZE, IMAGE_SIZE):
        rgb = cv2.resize(rgb, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
    return rgb


def load_mask(path: str) -> np.ndarray:
    if not path:
        return np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Cannot read mask: {path}")
    return cv2.resize(mask, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)


def make_overlay(original: np.ndarray, heatmap: np.ndarray, mask: np.ndarray) -> np.ndarray:
    overlay = cv2.addWeighted(original, ORIGINAL_WEIGHT, heatmap, HEATMAP_WEIGHT, 0)
    contours, _ = cv2.findContours(
        (mask > 0).astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    cv2.drawContours(overlay, contours, -1, (255, 255, 255), CONTOUR_PX)
    return overlay


def build_case(row: pd.Series, output_root: Path) -> Path:
    image_path = Path(row["image_path"])
    case_root = output_root / f"{row['category']}__{row['anomaly_type']}__{image_path.stem}"
    case_root.mkdir(parents=True, exist_ok=True)

    original = load_original(image_path)
    heatmap = load_heatmap(Path(row["heatmap_path"]))
    mask = load_mask(row["mask_path"])

    Image.fromarray(original).save(case_root / "original.png")
    Image.fromarray(mask).save(case_root / "ground_truth.png")
    Image.fromarray(heatmap).save(case_root / "heatmap.png")
    Image.fromarray(make_overlay(original, heatmap, mask)).save(case_root / "overlay.png")
    return case_root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    cases = pd.read_csv(args.cases, keep_default_na=False)
    for _, row in cases.iterrows():
        build_case(row, args.output)
    print(f"Built {len(cases)} cases: {args.output}")


if __name__ == "__main__":
    main()
