from __future__ import annotations

from pathlib import Path

from PIL import Image

from musc_demo.prepared_results import PreparedCatalog


FIXTURE_MANIFEST = Path(__file__).parent / "fixtures" / "prepared" / "manifest.json"


def populate_artifacts(root: Path) -> None:
    (root / "e01").mkdir(parents=True)
    (root / "e01" / "mvtec_ad_reproduced.csv").write_text(
        "category,auroc_sp\nscrew,82.31194916991187\n", encoding="utf-8"
    )
    (root / "e04q" / "screw-good-003").mkdir(parents=True)
    (root / "e04q" / "selected-qualitative-cases.csv").write_text(
        "category,relative_test_path\nscrew,good/003.png\n", encoding="utf-8"
    )
    for name, color in (
        ("original", (20, 40, 60)),
        ("ground_truth", (0, 0, 0)),
        ("heatmap", (255, 0, 0)),
        ("overlay", (100, 40, 60)),
    ):
        Image.new("RGB", (5, 4), color).save(
            root / "e04q" / "screw-good-003" / f"{name}.png"
        )


def load_catalog(tmp_path: Path) -> PreparedCatalog:
    artifact_root = tmp_path / "artifacts"
    populate_artifacts(artifact_root)
    return PreparedCatalog.load(FIXTURE_MANIFEST, artifact_root)


def test_catalog_lists_only_manifest_categories(tmp_path: Path) -> None:
    catalog = load_catalog(tmp_path)

    assert catalog.datasets() == ("mvtec_ad", "visa")
    assert catalog.categories("mvtec_ad") == ("screw",)
    assert catalog.categories("visa") == ("pcb1",)
    assert catalog.categories("unknown") == ()


def test_catalog_keeps_metrics_scores_and_assets_under_one_run_id(
    tmp_path: Path,
) -> None:
    catalog = load_catalog(tmp_path)
    run = catalog.get_run("E04Q-screw-top1")
    example = catalog.get_example("E04Q-screw-top1", "screw-good-003")

    assert run.experiment_id == "E04Q"
    assert run.metrics["top1_image_auroc"] == 72.59684361549498
    assert example.run_id == run.run_id
    assert example.raw_score == 0.585181798212395
    assert example.adjusted_score == 0.2828223705291748
    assert example.assets.heatmap is not None


def test_catalog_never_implicitly_combines_e01_metrics_with_e04q_assets(
    tmp_path: Path,
) -> None:
    catalog = load_catalog(tmp_path)
    e01 = catalog.get_run("E01-mvtec-baseline")
    e04q = catalog.get_run("E04Q-screw-top1")

    assert catalog.runs("mvtec_ad", "screw") == (
        "E01-mvtec-baseline",
        "E04Q-screw-top1",
    )
    assert e01.examples == ()
    assert "image_auroc" in e01.metrics
    assert "top1_image_auroc" not in e01.metrics
    assert "image_auroc" not in e04q.metrics


def test_metrics_preserve_recorded_names_and_values(tmp_path: Path) -> None:
    catalog = load_catalog(tmp_path)

    assert catalog.get_run("E01-mvtec-baseline").metrics == {
        "image_auroc": 82.31194916991187,
        "image_f1_max": 88.88888888888889,
        "image_ap": 90.95118443925398,
        "pixel_auroc": 98.76738203055838,
        "pixel_f1_max": 41.87712044390199,
        "pixel_ap": 36.11351775913153,
        "aupro": 94.40129356392521,
    }


def test_example_resolves_original_ground_truth_heatmap_and_overlay(
    tmp_path: Path,
) -> None:
    catalog = load_catalog(tmp_path)

    assets = catalog.get_example(
        "E04Q-screw-top1", "screw-good-003"
    ).assets

    assert assets.original is not None and assets.original.is_file()
    assert assets.ground_truth is not None and assets.ground_truth.is_file()
    assert assets.heatmap is not None and assets.heatmap.is_file()
    assert assets.overlay is not None and assets.overlay.is_file()


def test_missing_optional_asset_marks_only_that_asset_unavailable(
    tmp_path: Path,
) -> None:
    artifact_root = tmp_path / "artifacts"
    populate_artifacts(artifact_root)
    (artifact_root / "e04q" / "screw-good-003" / "ground_truth.png").unlink()

    catalog = PreparedCatalog.load(FIXTURE_MANIFEST, artifact_root)
    example = catalog.get_example("E04Q-screw-top1", "screw-good-003")

    assert catalog.get_run("E04Q-screw-top1").available is True
    assert example.assets.original is not None
    assert example.assets.ground_truth is None
    assert "ground_truth" in " ".join(example.notices)


def test_missing_category_artifacts_disable_only_that_category(
    tmp_path: Path,
) -> None:
    catalog = load_catalog(tmp_path)

    assert catalog.get_run("E01-mvtec-baseline").available is True
    assert catalog.get_run("E04Q-screw-top1").available is True
    assert catalog.get_run("E02-visa-pcb1").available is False
    assert "required artifact" in " ".join(
        catalog.get_run("E02-visa-pcb1").notices
    )


def test_corrupt_manifest_reports_a_catalog_error_without_crashing_app_state(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{not valid json", encoding="utf-8")

    catalog = PreparedCatalog.load(manifest, tmp_path / "artifacts")

    assert catalog.error is not None
    assert catalog.datasets() == ()


def test_prepared_entries_are_labeled_not_live_inference(tmp_path: Path) -> None:
    catalog = load_catalog(tmp_path)
    run = catalog.get_run("E04Q-screw-top1")
    example = catalog.get_example("E04Q-screw-top1", "screw-good-003")

    assert run.badge == "Prepared result - not live inference"
    assert example.badge == "Prepared result - not live inference"
