"""Deterministic downloads for one custom MuSc inference run."""

from __future__ import annotations

import csv
import json
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from .inference import InferenceResult
from .visualization import RenderedItem


@dataclass(frozen=True)
class ExportBundle:
    csv_path: Path
    manifest_path: Path
    zip_path: Path


def _ordered_unique(values: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _write_zip_member(
    archive: zipfile.ZipFile, source: Path, archive_name: str
) -> None:
    info = zipfile.ZipInfo(archive_name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    archive.writestr(info, source.read_bytes())


def export_run(
    result: InferenceResult,
    rendered: Sequence[RenderedItem],
    output_dir: Path,
) -> ExportBundle:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rendered_by_id = {item.image_id: item for item in rendered}
    if len(rendered_by_id) != len(rendered):
        raise ValueError("Rendered image IDs must be unique")

    ordered_items = sorted(
        result.items, key=lambda item: (-item.adjusted_score, item.image_id)
    )
    if {item.image_id for item in ordered_items} != set(rendered_by_id):
        raise ValueError("Rendered items do not match the inference result")

    csv_path = output_dir / "results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "rank",
                "image_id",
                "source_name",
                "raw_anomaly_score",
                "adjusted_anomaly_score",
            ]
        )
        for rank, item in enumerate(ordered_items, start=1):
            writer.writerow(
                [
                    rank,
                    item.image_id,
                    item.source_name,
                    item.raw_score,
                    item.adjusted_score,
                ]
            )

    render_warnings = [
        warning for item in rendered for warning in item.warnings
    ]
    warnings = _ordered_unique([*result.warnings, *render_warnings])
    manifest = {
        "run_type": "custom_upload",
        "upstream_commit": result.upstream_commit,
        "configuration": result.config,
        "rscin_neighbors": result.config.get("rscin_neighbors", [1, 2, 3]),
        "normalization_policy": "batch_global_minmax",
        "warnings": warnings,
        "images": [
            {
                "rank": rank,
                "image_id": item.image_id,
                "source_name": item.source_name,
                "raw_anomaly_score": item.raw_score,
                "adjusted_anomaly_score": item.adjusted_score,
                "heatmap": f"heatmaps/{item.image_id}.png",
                "overlay": f"overlays/{item.image_id}.png",
            }
            for rank, item in enumerate(ordered_items, start=1)
        ],
    }
    manifest_path = output_dir / "run-manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    zip_path = output_dir / "musc-demo-results.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        for directory, attribute in (
            ("heatmaps", "heatmap_path"),
            ("overlays", "overlay_path"),
        ):
            for item in sorted(rendered, key=lambda candidate: candidate.image_id):
                _write_zip_member(
                    archive,
                    getattr(item, attribute),
                    f"{directory}/{item.image_id}.png",
                )
        _write_zip_member(archive, csv_path, "results.csv")
        _write_zip_member(archive, manifest_path, "run-manifest.json")

    return ExportBundle(
        csv_path=csv_path,
        manifest_path=manifest_path,
        zip_path=zip_path,
    )
