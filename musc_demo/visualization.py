"""Batch-consistent anomaly heatmaps and overlays."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from .inference import InferenceResult, ProgressCallback


@dataclass(frozen=True)
class NormalizedMaps:
    maps: np.ndarray
    minimum: float
    maximum: float
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class RenderedItem:
    image_id: str
    source_name: str
    original_path: Path
    heatmap_path: Path
    overlay_path: Path
    raw_score: float
    adjusted_score: float
    warnings: tuple[str, ...]


def normalize_batch_maps(maps: np.ndarray) -> NormalizedMaps:
    values = np.asarray(maps, dtype=np.float32)
    if values.ndim != 3 or values.shape[0] == 0:
        raise ValueError("Anomaly maps must have shape [N, H, W]")
    if not np.isfinite(values).all():
        raise ValueError("Anomaly maps must contain only finite values")

    minimum = float(values.min())
    maximum = float(values.max())
    value_range = maximum - minimum
    if value_range <= np.finfo(np.float32).eps:
        return NormalizedMaps(
            maps=np.zeros_like(values),
            minimum=minimum,
            maximum=maximum,
            warnings=("CONSTANT_ANOMALY_MAPS",),
        )
    return NormalizedMaps(
        maps=(values - minimum) / value_range,
        minimum=minimum,
        maximum=maximum,
        warnings=(),
    )


def _jet_rgb(normalized_map: np.ndarray) -> np.ndarray:
    values = np.clip(normalized_map, 0.0, 1.0)
    red = np.clip(1.5 - np.abs(4.0 * values - 3.0), 0.0, 1.0)
    green = np.clip(1.5 - np.abs(4.0 * values - 2.0), 0.0, 1.0)
    blue = np.clip(1.5 - np.abs(4.0 * values - 1.0), 0.0, 1.0)
    return np.rint(np.stack((red, green, blue), axis=-1) * 255.0).astype(
        np.uint8
    )


def render_result_images(
    result: InferenceResult,
    output_dir: Path,
    progress: ProgressCallback,
) -> tuple[RenderedItem, ...]:
    progress("render", 0.95, "Tạo heatmap và overlay")
    output_dir = Path(output_dir)
    heatmap_dir = output_dir / "heatmaps"
    overlay_dir = output_dir / "overlays"
    heatmap_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)

    normalized = normalize_batch_maps(
        np.stack([item.anomaly_map for item in result.items])
    )
    rendered: list[RenderedItem] = []
    for index, item in enumerate(result.items):
        with Image.open(item.path) as source:
            original = ImageOps.exif_transpose(source).convert("RGB")
        heatmap = Image.fromarray(_jet_rgb(normalized.maps[index]), mode="RGB")
        heatmap = heatmap.resize(original.size, resample=Image.Resampling.BILINEAR)
        overlay = Image.blend(original, heatmap, alpha=0.45)
        heatmap_path = heatmap_dir / f"{item.image_id}.png"
        overlay_path = overlay_dir / f"{item.image_id}.png"
        heatmap.save(heatmap_path, format="PNG")
        overlay.save(overlay_path, format="PNG")
        rendered.append(
            RenderedItem(
                image_id=item.image_id,
                source_name=item.source_name,
                original_path=item.path,
                heatmap_path=heatmap_path,
                overlay_path=overlay_path,
                raw_score=item.raw_score,
                adjusted_score=item.adjusted_score,
                warnings=normalized.warnings,
            )
        )
    return tuple(rendered)

