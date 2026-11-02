#!/usr/bin/env python3
"""Summarize the E04 pooling comparison over all 15 MVTec AD categories.

--results-root holds one directory per category, each being the OUTPUT_ROOT
passed to scripts/run_category.sh with patches/e04-top1-pooling.patch applied.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

CATEGORIES = (
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill", "screw",
    "tile", "toothbrush", "transistor", "wood", "zipper",
)
VARIANTS = ("max_raw", "max_rscin", "top1_raw", "top1_rscin")
METRICS = ("auroc_sp", "f1_sp", "ap_sp")
CONTEXTS = (
    ("without_rscin", "top1_raw", "max_raw"),
    ("with_rscin", "top1_rscin", "max_rscin"),
)
TIE_TOLERANCE = 1e-9
RESULT_SUBDIR = Path("mvtec_ad") / "ViT-L-14-336" / "imagesize518"


def load_results(results_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows = []
    score_frames = []
    for category in CATEGORIES:
        result_root = results_root / category / RESULT_SUBDIR
        payload = json.loads((result_root / "e04-metrics.json").read_text(encoding="utf-8"))
        for variant, values in payload["variants"].items():
            metric_rows.append({"category": category, "variant": variant, **values})
        scores = pd.read_csv(result_root / "e04-per-image-scores.csv")
        score_frames.append(scores.assign(category=category))
    return pd.DataFrame(metric_rows), pd.concat(score_frames, ignore_index=True)


def summarize_macro(variant_metrics: pd.DataFrame) -> pd.DataFrame:
    return (
        variant_metrics.groupby("variant", sort=False)[list(METRICS)]
        .mean()
        .reindex(list(VARIANTS))
        .reset_index()
    )


def compute_pooling_deltas(variant_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for category in CATEGORIES:
        current = variant_metrics[variant_metrics["category"] == category].set_index("variant")
        row: dict[str, object] = {"category": category}
        for metric in METRICS:
            row[f"{metric}_top1_minus_max_raw"] = (
                current.loc["top1_raw", metric] - current.loc["max_raw", metric]
            )
            row[f"{metric}_top1_minus_max_rscin"] = (
                current.loc["top1_rscin", metric] - current.loc["max_rscin", metric]
            )
        rows.append(row)
    deltas = pd.DataFrame(rows)
    mean_row = {"category": "mean", **deltas.drop(columns="category").mean().to_dict()}
    return pd.concat([deltas, pd.DataFrame([mean_row])], ignore_index=True)


def compute_win_tie_loss(variant_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for context, top_variant, max_variant in CONTEXTS:
        for metric in METRICS:
            paired = variant_metrics.pivot(index="category", columns="variant", values=metric)
            delta = paired[top_variant] - paired[max_variant]
            rows.append({
                "context": context,
                "metric": metric,
                "wins": int((delta > TIE_TOLERANCE).sum()),
                "ties": int((delta.abs() <= TIE_TOLERANCE).sum()),
                "losses": int((delta < -TIE_TOLERANCE).sum()),
                "mean_delta": float(delta.mean()),
                "median_delta": float(delta.median()),
            })
    return pd.DataFrame(rows)


def best_f1_operating_point(labels: pd.Series, scores: pd.Series) -> dict[str, float | int]:
    """Return the benchmark-optimized threshold, which is post hoc by construction."""
    label_values = np.asarray(labels, dtype=np.int32)
    score_values = np.asarray(scores, dtype=np.float64)
    precisions, recalls, thresholds = precision_recall_curve(label_values, score_values)
    denominators = precisions + recalls
    f1_values = np.divide(
        2 * precisions * recalls,
        denominators,
        out=np.zeros_like(denominators),
        where=denominators != 0,
    )
    best_index = int(np.nanargmax(f1_values))
    if best_index >= len(thresholds):
        raise ValueError("F1-max has no finite threshold")
    threshold = float(thresholds[best_index])
    predictions = (score_values >= threshold).astype(np.int32)
    return {
        "threshold": threshold,
        "f1": float(f1_values[best_index]),
        "tp": int(((predictions == 1) & (label_values == 1)).sum()),
        "fp": int(((predictions == 1) & (label_values == 0)).sum()),
        "tn": int(((predictions == 0) & (label_values == 0)).sum()),
        "fn": int(((predictions == 0) & (label_values == 1)).sum()),
    }


def compute_operating_points(scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for category in CATEGORIES:
        category_scores = scores[scores["category"] == category]
        for variant in VARIANTS:
            rows.append({
                "category": category,
                "variant": variant,
                **best_f1_operating_point(category_scores["label"], category_scores[variant]),
            })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    variant_metrics, scores = load_results(args.results_root)
    args.output.mkdir(parents=True, exist_ok=True)
    macro_summary = summarize_macro(variant_metrics)
    win_tie_loss = compute_win_tie_loss(variant_metrics)

    variant_metrics.to_csv(args.output / "e04-category-metrics-long.csv", index=False)
    macro_summary.to_csv(args.output / "e04-macro-summary.csv", index=False)
    compute_pooling_deltas(variant_metrics).to_csv(args.output / "e04-pooling-deltas.csv", index=False)
    win_tie_loss.to_csv(args.output / "e04-win-tie-loss.csv", index=False)
    compute_operating_points(scores).to_csv(args.output / "e04-operating-points.csv", index=False)

    print(macro_summary.to_string(index=False))
    print(win_tie_loss.to_string(index=False))


if __name__ == "__main__":
    main()
