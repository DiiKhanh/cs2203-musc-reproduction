from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from musc_demo.inference import InferenceItem, InferenceResult
from musc_demo.visualization import normalize_batch_maps, render_result_images


def make_result(tmp_path: Path) -> InferenceResult:
    first = tmp_path / "a" / "image.png"
    second = tmp_path / "b" / "image.png"
    first.parent.mkdir()
    second.parent.mkdir()
    Image.new("RGB", (9, 5), (20, 40, 60)).save(first)
    Image.new("RGB", (7, 11), (60, 40, 20)).save(second)
    return InferenceResult(
        items=(
            InferenceItem(
                image_id="first-id",
                source_name="a/image.png",
                path=first,
                raw_score=4.0,
                adjusted_score=0.25,
                anomaly_map=np.asarray([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32),
            ),
            InferenceItem(
                image_id="second-id",
                source_name="b/image.png",
                path=second,
                raw_score=8.0,
                adjusted_score=0.75,
                anomaly_map=np.asarray([[4.0, 5.0], [6.0, 7.0]], dtype=np.float32),
            ),
        ),
        warnings=(),
        config={"rscin_neighbors": [1, 2, 3]},
        upstream_commit="pinned-commit",
    )


def test_batch_wide_normalization_uses_one_global_range() -> None:
    maps = np.asarray(
        [
            [[0.0, 1.0], [2.0, 3.0]],
            [[4.0, 5.0], [6.0, 7.0]],
        ],
        dtype=np.float32,
    )

    normalized = normalize_batch_maps(maps)

    assert normalized.minimum == 0.0
    assert normalized.maximum == 7.0
    assert normalized.warnings == ()
    np.testing.assert_allclose(normalized.maps[0, 1, 1], 3.0 / 7.0)
    np.testing.assert_allclose(normalized.maps[1, 0, 0], 4.0 / 7.0)


def test_constant_maps_render_as_zero_and_warn() -> None:
    maps = np.full((2, 3, 3), 5.0, dtype=np.float32)

    normalized = normalize_batch_maps(maps)

    np.testing.assert_array_equal(normalized.maps, np.zeros_like(maps))
    assert normalized.warnings == ("CONSTANT_ANOMALY_MAPS",)


def test_render_emits_the_render_progress_stage(tmp_path: Path) -> None:
    stages: list[str] = []

    render_result_images(
        make_result(tmp_path),
        tmp_path / "output",
        progress=lambda stage, fraction, message: stages.append(stage),
    )

    assert stages == ["render"]


def test_overlay_preserves_original_dimensions(tmp_path: Path) -> None:
    rendered = render_result_images(
        make_result(tmp_path), tmp_path / "output", progress=lambda *_: None
    )

    with Image.open(rendered[0].overlay_path) as first_overlay:
        assert first_overlay.size == (9, 5)
    with Image.open(rendered[1].overlay_path) as second_overlay:
        assert second_overlay.size == (7, 11)


def test_unique_ids_prevent_filename_overwrite(tmp_path: Path) -> None:
    rendered = render_result_images(
        make_result(tmp_path), tmp_path / "output", progress=lambda *_: None
    )

    assert {item.heatmap_path.name for item in rendered} == {
        "first-id.png",
        "second-id.png",
    }
    assert {item.overlay_path.name for item in rendered} == {
        "first-id.png",
        "second-id.png",
    }
    assert all(item.heatmap_path.is_file() for item in rendered)
    assert all(item.overlay_path.is_file() for item in rendered)
