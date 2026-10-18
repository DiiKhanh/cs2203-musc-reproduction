from __future__ import annotations

import csv
import json
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from musc_demo.export import export_run
from musc_demo.inference import InferenceItem, InferenceResult
from musc_demo.visualization import render_result_images


def make_export_inputs(tmp_path: Path):
    first = tmp_path / "source-a" / "image.png"
    second = tmp_path / "source-b" / "image.png"
    first.parent.mkdir()
    second.parent.mkdir()
    Image.new("RGB", (6, 4), (10, 20, 30)).save(first)
    Image.new("RGB", (5, 7), (30, 20, 10)).save(second)
    result = InferenceResult(
        items=(
            InferenceItem(
                image_id="z-low",
                source_name="source-a/image.png",
                path=first,
                raw_score=2.0,
                adjusted_score=0.2,
                anomaly_map=np.asarray([[0.0, 1.0], [1.0, 2.0]], dtype=np.float32),
            ),
            InferenceItem(
                image_id="a-high",
                source_name="source-b/image.png",
                path=second,
                raw_score=9.0,
                adjusted_score=0.9,
                anomaly_map=np.asarray([[3.0, 4.0], [4.0, 5.0]], dtype=np.float32),
            ),
        ),
        warnings=("EXAMPLE_WARNING",),
        config={
            "image_size": 518,
            "feature_layers": [5, 11, 17, 23],
            "rscin_neighbors": [1, 2, 3],
        },
        upstream_commit="pinned-commit",
    )
    rendered = render_result_images(
        result, tmp_path / "run", progress=lambda *_: None
    )
    return result, rendered


def test_export_csv_matches_adjusted_score_order(tmp_path: Path) -> None:
    result, rendered = make_export_inputs(tmp_path)

    bundle = export_run(result, rendered, tmp_path / "run")

    with bundle.csv_path.open(newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))
    assert [row["image_id"] for row in rows] == ["a-high", "z-low"]
    assert [row["rank"] for row in rows] == ["1", "2"]
    assert [row["adjusted_anomaly_score"] for row in rows] == ["0.9", "0.2"]


def test_export_zip_contains_manifest_csv_heatmaps_and_overlays(
    tmp_path: Path,
) -> None:
    result, rendered = make_export_inputs(tmp_path)

    bundle = export_run(result, rendered, tmp_path / "run")

    with zipfile.ZipFile(bundle.zip_path) as archive:
        assert archive.namelist() == [
            "heatmaps/a-high.png",
            "heatmaps/z-low.png",
            "overlays/a-high.png",
            "overlays/z-low.png",
            "results.csv",
            "run-manifest.json",
        ]
        assert archive.testzip() is None


def test_export_manifest_records_config_commit_and_normalization_policy(
    tmp_path: Path,
) -> None:
    result, rendered = make_export_inputs(tmp_path)

    bundle = export_run(result, rendered, tmp_path / "run")

    manifest = json.loads(bundle.manifest_path.read_text(encoding="utf-8"))
    assert manifest["run_type"] == "custom_upload"
    assert manifest["upstream_commit"] == "pinned-commit"
    assert manifest["configuration"]["image_size"] == 518
    assert manifest["rscin_neighbors"] == [1, 2, 3]
    assert manifest["normalization_policy"] == "batch_global_minmax"
    assert manifest["warnings"] == ["EXAMPLE_WARNING"]


def test_export_contains_no_labels_or_metrics_for_custom_upload(
    tmp_path: Path,
) -> None:
    result, rendered = make_export_inputs(tmp_path)

    bundle = export_run(result, rendered, tmp_path / "run")

    with bundle.csv_path.open(newline="", encoding="utf-8") as csv_file:
        header = next(csv.reader(csv_file))
    manifest = json.loads(bundle.manifest_path.read_text(encoding="utf-8"))
    serialized = json.dumps(manifest).lower()
    assert header == [
        "rank",
        "image_id",
        "source_name",
        "raw_anomaly_score",
        "adjusted_anomaly_score",
    ]
    assert "label" not in serialized
    assert "ground_truth" not in serialized
    assert "metrics" not in serialized
