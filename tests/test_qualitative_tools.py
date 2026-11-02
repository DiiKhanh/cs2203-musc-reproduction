"""Tests for the qualitative and E04 post-processing tools."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import build_qualitative_assets as assets  # noqa: E402
import render_slide_strips as strips  # noqa: E402
import select_qualitative_cases as selection  # noqa: E402
import summarize_e04 as e04  # noqa: E402

VISA_CLASS_ID = 5


def write_case(root: Path, mask: np.ndarray) -> None:
    size = mask.shape[0]
    original = np.full((size, size, 3), 90, dtype=np.uint8)
    heatmap = np.full((size, size, 3), 30, dtype=np.uint8)
    Image.fromarray(original).save(root / "original.png")
    Image.fromarray(heatmap).save(root / "heatmap.png")
    Image.fromarray(mask).save(root / "ground_truth.png")
    Image.fromarray(assets.make_overlay(original, heatmap, mask)).save(root / "overlay.png")


def test_visa_class_id_mask_renders_white(tmp_path):
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[20:30, 20:30] = VISA_CLASS_ID
    write_case(tmp_path, mask)

    strip, mask_pixels, values = strips.render_case(tmp_path)

    mask_column = strip[:, 64 + strips.GAP_PX:2 * 64 + strips.GAP_PX]
    assert mask_pixels == 100
    assert values == [0, VISA_CLASS_ID]
    assert mask_column[25, 25].tolist() == [255, 255, 255]
    assert mask_column[0, 0].tolist() == [0, 0, 0]


def test_contour_uses_every_nonzero_class_id():
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[20:30, 20:30] = VISA_CLASS_ID
    blank = np.zeros((64, 64, 3), dtype=np.uint8)

    overlay = assets.make_overlay(blank, blank, mask)

    assert overlay[20, 25].tolist() == [255, 255, 255]


def test_empty_mask_path_gives_an_empty_mask():
    assert not assets.load_mask("").any()


def make_e02q_rows(category: str) -> pd.DataFrame:
    return pd.DataFrame({
        "category": category,
        "image_path": [f"{category}/{index}" for index in range(8)],
        "is_anomaly": [1, 1, 1, 1, 0, 0, 0, 0],
        "rscin_score": [0.1, 0.2, 0.8, 0.9, 0.7, 0.6, 0.2, 0.1],
        "pixel_auroc_raw": [0.99, 0.5, 0.7, 0.6, np.nan, np.nan, np.nan, np.nan],
    })


def test_e02q_selection_keeps_cases_unique_within_a_category():
    rows = pd.concat([make_e02q_rows(name) for name in selection.E02Q_CATEGORIES])

    selected = selection.select_e02q(rows)

    assert len(selected) == 24
    assert selected["image_path"].is_unique
    first = selected[selected["category"] == "pcb2"].set_index("selection_reason")
    # Images 0 and 1 are taken as low-score anomalies, so localization skips them.
    assert first.loc["best_localization", "image_path"] == "pcb2/2"
    assert first.loc["worst_localization", "image_path"] == "pcb2/3"


def test_selection_fails_when_too_few_cases_remain():
    rows = make_e02q_rows("pcb2").head(1)

    with pytest.raises(ValueError, match="cases available"):
        selection.take_unique(rows, "rscin_score", 2, True, "reason", set())


def test_win_tie_loss_counts_tied_categories():
    rows = []
    for index, category in enumerate(e04.CATEGORIES):
        for variant in e04.VARIANTS:
            gain = 1.0 if variant == "top1_rscin" and index == 0 else 0.0
            rows.append({
                "category": category,
                "variant": variant,
                "auroc_sp": 90.0 + gain,
                "f1_sp": 90.0,
                "ap_sp": 90.0,
            })

    table = e04.compute_win_tie_loss(pd.DataFrame(rows))

    row = table[(table["context"] == "with_rscin") & (table["metric"] == "auroc_sp")].iloc[0]
    assert (row["wins"], row["ties"], row["losses"]) == (1, 14, 0)
    assert row["median_delta"] == 0


def test_operating_point_counts_confusion_at_the_best_threshold():
    labels = pd.Series([0, 0, 1, 1])
    scores = pd.Series([0.1, 0.2, 0.8, 0.9])

    point = e04.best_f1_operating_point(labels, scores)

    assert (point["tp"], point["fp"], point["tn"], point["fn"]) == (2, 0, 2, 0)
    assert point["f1"] == 1
