#!/usr/bin/env python3
"""Select the qualitative cases reported for E01Q, E02Q, and E04Q.

The output CSV lists image, mask, and heatmap paths for
tools/build_qualitative_assets.py. --results-root holds one directory per
category, each being the OUTPUT_ROOT passed to scripts/run_category.sh.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

E01Q_CATEGORIES = ("leather", "transistor", "screw")
E02Q_CATEGORIES = ("pcb2", "pcb3", "pcb4", "fryum")
RESULT_SUBDIR = Path("ViT-L-14-336") / "imagesize518"
TOP_RESPONSE_FRACTION = 0.01
HEATMAP_SUFFIXES = {".png", ".jpg", ".jpeg"}


def result_dir(results_root: Path, category: str, dataset: str) -> Path:
    return results_root / category / dataset / RESULT_SUBDIR


def find_mvtec_mask(data_root: Path, category: str, anomaly_type: str, stem: str) -> Path:
    candidates = sorted((data_root / category / "ground_truth" / anomaly_type).glob(f"{stem}*"))
    if len(candidates) != 1:
        raise ValueError(f"Expected one mask for {category}/{anomaly_type}/{stem}: {candidates}")
    return candidates[0]


def build_jet_inverse() -> np.ndarray:
    jet_lut = cv2.applyColorMap(
        np.arange(256, dtype=np.uint8).reshape(256, 1),
        cv2.COLORMAP_JET,
    ).reshape(256, 3)
    keys = (
        jet_lut[:, 0].astype(np.uint32)
        | (jet_lut[:, 1].astype(np.uint32) << 8)
        | (jet_lut[:, 2].astype(np.uint32) << 16)
    )
    inverse = np.full(1 << 24, -1, dtype=np.int16)
    inverse[keys] = np.arange(256, dtype=np.int16)
    return inverse


def heatmap_to_score(path: Path, inverse: np.ndarray) -> np.ndarray:
    """Recover the 8-bit score from a lossless JET heatmap saved by MuSc."""
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Cannot read heatmap: {path}")
    keys = (
        bgr[:, :, 0].astype(np.uint32)
        | (bgr[:, :, 1].astype(np.uint32) << 8)
        | (bgr[:, :, 2].astype(np.uint32) << 16)
    )
    score = inverse[keys]
    if not np.all(score >= 0):
        raise ValueError(f"Heatmap contains a colour outside the JET table: {path}")
    return score.astype(np.float32) / 255.0


def score_e01q_image(heatmap_path: Path, mask_path: Path | None, inverse: np.ndarray) -> tuple[float, float]:
    score = heatmap_to_score(heatmap_path, inverse)
    flat = score.reshape(-1)
    top_count = max(1, int(np.ceil(flat.size * TOP_RESPONSE_FRACTION)))
    top_mean = float(np.partition(flat, -top_count)[-top_count:].mean())
    if mask_path is None:
        return float("nan"), top_mean

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Cannot read mask: {mask_path}")
    mask = cv2.resize(mask, (score.shape[1], score.shape[0]), interpolation=cv2.INTER_NEAREST)
    return float(roc_auc_score((mask > 0).astype(np.uint8).reshape(-1), flat)), top_mean


def score_e01q_cases(results_root: Path, data_root: Path) -> pd.DataFrame:
    inverse = build_jet_inverse()
    records = []
    for category in E01Q_CATEGORIES:
        heatmap_root = result_dir(results_root, category, "mvtec_ad") / category
        heatmaps = sorted(
            path for path in heatmap_root.rglob("*")
            if path.is_file() and path.suffix.lower() in HEATMAP_SUFFIXES
        )
        for heatmap_path in heatmaps:
            anomaly_type = heatmap_path.parent.name
            mask_path = None
            if anomaly_type != "good":
                mask_path = find_mvtec_mask(data_root, category, anomaly_type, heatmap_path.stem)
            pixel_auroc, top_mean = score_e01q_image(heatmap_path, mask_path, inverse)
            records.append({
                "category": category,
                "anomaly_type": anomaly_type,
                "image_name": heatmap_path.name,
                "image_path": str(data_root / category / "test" / anomaly_type / heatmap_path.name),
                "mask_path": str(mask_path) if mask_path else "",
                "heatmap_path": str(heatmap_path),
                "pixel_auroc_from_saved_heatmap": pixel_auroc,
                "top1_percent_heatmap_mean": top_mean,
            })
    return pd.DataFrame(records)


def select_e01q(case_scores: pd.DataFrame) -> pd.DataFrame:
    selected_parts = []
    for category in E01Q_CATEGORIES:
        rows = case_scores[case_scores["category"] == category]
        anomaly_rows = rows[rows["anomaly_type"] != "good"]
        normal_rows = rows[rows["anomaly_type"] == "good"]
        for chosen, reason in (
            (anomaly_rows.nlargest(2, "pixel_auroc_from_saved_heatmap"), "best_localization"),
            (anomaly_rows.nsmallest(2, "pixel_auroc_from_saved_heatmap"), "worst_localization"),
            (normal_rows.nlargest(2, "top1_percent_heatmap_mean"), "high_response_normal_candidate"),
        ):
            selected_parts.append(chosen.assign(selection_reason=reason))
    selected = pd.concat(selected_parts, ignore_index=True)
    return selected.drop_duplicates(subset=["category", "anomaly_type", "image_name"], keep="first")


def score_e02q_cases(results_root: Path, data_root: Path) -> pd.DataFrame:
    meta = json.loads((data_root / "meta.json").read_text(encoding="utf-8"))
    frames = []
    for category in E02Q_CATEGORIES:
        result_root = result_dir(results_root, category, "visa")
        scores = pd.read_csv(result_root / f"{category}_per-image-scores.csv")
        mask_by_image = {
            str(data_root / sample["img_path"]): sample["mask_path"]
            for sample in meta["test"][category]
        }
        if set(mask_by_image) != set(scores["image_path"]):
            raise ValueError(f"meta.json and the score table disagree for {category}")

        image_paths = scores["image_path"].map(Path)
        anomaly_types = image_paths.map(lambda path: path.parent.name)
        frames.append(scores.assign(
            anomaly_type=anomaly_types,
            mask_path=scores["image_path"].map(
                lambda path: str(data_root / mask_by_image[path]) if mask_by_image[path] else ""
            ),
            heatmap_path=[
                str(result_root / category / kind / path.name)
                for kind, path in zip(anomaly_types, image_paths)
            ],
        ))
    return pd.concat(frames, ignore_index=True)


def take_unique(
    rows: pd.DataFrame,
    column: str,
    count: int,
    ascending: bool,
    reason: str,
    used_paths: set[str],
) -> pd.DataFrame:
    # The image path breaks score ties so the selection is deterministic.
    candidates = rows.sort_values([column, "image_path"], ascending=[ascending, True], kind="mergesort")
    chosen = candidates[~candidates["image_path"].isin(used_paths)].head(count)
    if len(chosen) != count:
        raise ValueError(f"Only {len(chosen)} of {count} cases available for {reason}")
    used_paths.update(chosen["image_path"])
    return chosen.assign(selection_reason=reason)


def select_e02q(case_scores: pd.DataFrame) -> pd.DataFrame:
    selected_parts = []
    for category in E02Q_CATEGORIES:
        rows = case_scores[case_scores["category"] == category]
        anomaly_rows = rows[rows["is_anomaly"].astype(int) == 1]
        normal_rows = rows[rows["is_anomaly"].astype(int) == 0]
        used_paths: set[str] = set()
        selected_parts.extend([
            take_unique(anomaly_rows, "rscin_score", 2, True, "false_negative_candidate", used_paths),
            take_unique(normal_rows, "rscin_score", 2, False, "false_positive_candidate", used_paths),
            take_unique(anomaly_rows, "pixel_auroc_raw", 1, False, "best_localization", used_paths),
            take_unique(anomaly_rows, "pixel_auroc_raw", 1, True, "worst_localization", used_paths),
        ])
    return pd.concat(selected_parts, ignore_index=True)


def locate_e04q_cases(results_root: Path, data_root: Path, cases_path: Path) -> pd.DataFrame:
    records = []
    for plan in pd.read_csv(cases_path).to_dict("records"):
        category = plan["category"]
        anomaly_type, image_name = plan["relative_test_path"].split("/", 1)
        mask_path = ""
        if int(plan["expected_label"]) == 1:
            stem = Path(image_name).stem
            mask_path = str(find_mvtec_mask(data_root, category, anomaly_type, stem))
        records.append({
            "category": category,
            "anomaly_type": anomaly_type,
            "image_path": str(data_root / category / "test" / anomaly_type / image_name),
            "mask_path": mask_path,
            "heatmap_path": str(
                result_dir(results_root, category, "mvtec_ad") / category / anomaly_type / image_name
            ),
            "selection_reason": plan["selection_reason"],
        })
    return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=("e01q", "e02q", "e04q"), required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", type=Path, help="Fixed case plan, required for e04q")
    args = parser.parse_args()

    if args.experiment == "e01q":
        selected = select_e01q(score_e01q_cases(args.results_root, args.data_root))
    elif args.experiment == "e02q":
        selected = select_e02q(score_e02q_cases(args.results_root, args.data_root))
    else:
        if args.cases is None:
            parser.error("--cases is required for e04q")
        selected = locate_e04q_cases(args.results_root, args.data_root, args.cases)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(args.output, index=False)
    print(f"Selected {len(selected)} cases: {args.output}")


if __name__ == "__main__":
    main()
