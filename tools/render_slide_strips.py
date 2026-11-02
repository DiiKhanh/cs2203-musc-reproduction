"""Render slide-sized four-image strips from stored MuSc qualitative assets.

Heatmaps and originals are reused byte for byte. The ground truth is shown as a
binary mask because VisA masks store defect class IDs (1-7) rather than 255.

--source is a directory holding the E01Q and E02Q run ZIPs extracted into
subdirectories named e01q and e02q. --output is the figure directory to fill.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

GAP_PX = 24
CONTOUR_INNER_PX = 2
CONTOUR_OUTER_PX = 5
CASES = {
    "e01q-leather-glue-011-slide.png": "e01q/selected-assets/leather__glue__011",
    "e02q-pcb4-anomaly-032-slide.png": "e02q/selected-assets/pcb4__Anomaly__032",
    "e02q-pcb3-anomaly-043-slide.png": "e02q/selected-assets/pcb3__Anomaly__043",
    "e01q-transistor-cut-lead-008-slide.png": "e01q/selected-assets/transistor__cut_lead__008",
    "e02q-fryum-anomaly-010-slide.png": "e02q/selected-assets/fryum__Anomaly__010",
}


def load_rgb(path):
    return np.asarray(Image.open(path).convert("RGB"))


def render_case(case_root):
    original = load_rgb(case_root / "original.png")
    heatmap = load_rgb(case_root / "heatmap.png")
    stored_overlay = load_rgb(case_root / "overlay.png")
    raw_mask = np.asarray(Image.open(case_root / "ground_truth.png"))
    binary = (raw_mask > 0).astype(np.uint8)
    assert binary.any(), f"Empty mask: {case_root}"

    blended = cv2.addWeighted(original, 0.55, heatmap, 0.45, 0)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # The stored overlay must equal the blend wherever the old contour was not drawn.
    # One grey level of tolerance covers rounding differences between OpenCV releases.
    contour_zone = np.zeros(binary.shape, dtype=np.uint8)
    cv2.drawContours(contour_zone, contours, -1, 1, 6)
    difference = np.abs(blended.astype(int) - stored_overlay.astype(int)).max(-1)
    assert difference[contour_zone == 0].max() <= 1, f"Overlay drift: {case_root}"

    # A dark halo keeps the white contour visible on pale objects such as fryum.
    overlay = blended.copy()
    cv2.drawContours(overlay, contours, -1, (0, 0, 0), CONTOUR_OUTER_PX, cv2.LINE_AA)
    cv2.drawContours(overlay, contours, -1, (255, 255, 255), CONTOUR_INNER_PX, cv2.LINE_AA)

    mask_rgb = np.repeat((binary * 255)[..., None], 3, axis=-1)
    height = original.shape[0]
    gap = np.full((height, GAP_PX, 3), 247, dtype=np.uint8)
    gap[..., 1] = 249
    gap[..., 2] = 252
    strip = np.concatenate([original, gap, mask_rgb, gap, heatmap, gap, overlay], axis=1)
    return strip, int(binary.sum()), np.unique(raw_mask).tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    for name, relative in CASES.items():
        strip, mask_pixels, values = render_case(args.source / relative)
        Image.fromarray(strip).save(args.output / name, optimize=True)
        print(f"{name}: {strip.shape[1]}x{strip.shape[0]}, mask px={mask_pixels}, raw values={values}")


if __name__ == "__main__":
    main()
