from __future__ import annotations

import re
import socket
import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from playwright.sync_api import Page, expect, sync_playwright

from musc_demo.app import DemoServices, build_app, demo_theme, theme_css_path
from musc_demo.inference import InferenceItem, InferenceResult
from musc_demo.prepared_results import (
    PreparedAssets,
    PreparedCatalog,
    PreparedExample,
    PreparedRun,
)


class FakeEngine:
    def __init__(self) -> None:
        self.state = "uninitialized"

    @property
    def preprocess(self):
        return lambda image: image

    def initialize(self) -> None:
        self.state = "ready"

    def run(self, dataset, progress) -> InferenceResult:
        for stage, fraction, message in (
            ("normalize", 0.05, "Chuẩn hóa ảnh"),
            ("clip", 0.20, "Trích xuất CLIP features"),
            ("lnamd", 0.45, "Tính LNAMD"),
            ("msm", 0.75, "Tính mutual scoring"),
            ("rscin", 0.90, "Điều chỉnh anomaly score bằng RsCIN"),
        ):
            progress(stage, fraction, message)
            time.sleep(0.6)
        items = tuple(
            InferenceItem(
                image_id=image.image_id,
                source_name=image.source_name,
                path=image.path,
                raw_score=float(index + 1),
                adjusted_score=float(index + 1) / len(dataset.images),
                anomaly_map=np.full((12, 12), index / len(dataset.images)),
            )
            for index, image in enumerate(dataset.images)
        )
        return InferenceResult(
            items=items,
            warnings=(),
            config={"rscin_neighbors": [1, 2, 3]},
            upstream_commit="fake-browser-test",
        )


def _image(path: Path, color: tuple[int, int, int]) -> Path:
    Image.new("RGB", (96, 72), color).save(path)
    return path


def _prepared_catalog(root: Path) -> PreparedCatalog:
    original = _image(root / "prepared-original.png", (65, 90, 120))
    ground_truth = _image(root / "prepared-ground-truth.png", (0, 0, 0))
    heatmap = _image(root / "prepared-heatmap.png", (220, 45, 35))
    overlay = _image(root / "prepared-overlay.png", (145, 65, 70))
    example = PreparedExample(
        run_id="mvtec-bottle-e01",
        image_id="bottle-001",
        source_name="broken_large/001.png",
        defect_type="broken_large",
        role="success",
        raw_score=0.71,
        adjusted_score=0.82,
        assets=PreparedAssets(original, ground_truth, heatmap, overlay),
        notices=(),
    )
    run = PreparedRun(
        run_id="mvtec-bottle-e01",
        experiment_id="E01",
        dataset="MVTec AD",
        category="bottle",
        timestamp="2026-10-18T00:00:00Z",
        source_commit="fake-browser-test",
        description="Browser E2E prepared result",
        sample_counts={"normal": 20, "anomalous": 63},
        defect_types=("broken_large",),
        ground_truth_available=True,
        metrics={"Image AUROC": 0.99},
        examples=(example,),
        available=True,
        notices=(),
    )
    return PreparedCatalog((run,))


def _free_port() -> int:
    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        return int(server.getsockname()[1])


@pytest.fixture(scope="module")
def browser_demo(tmp_path_factory):
    root = tmp_path_factory.mktemp("browser-demo")
    services = DemoServices(
        engine=FakeEngine(),
        prepared=_prepared_catalog(root),
        workspace_root=root / "runs",
        gpu_available=True,
        gpu_name="Fake Tesla T4",
    )
    app = build_app(services)
    _, local_url, _ = app.launch(
        server_name="127.0.0.1",
        server_port=_free_port(),
        prevent_thread_lock=True,
        quiet=True,
        show_error=False,
        css_paths=theme_css_path(),
        theme=demo_theme(),
    )
    try:
        yield local_url, root
    finally:
        app.close()


def _no_horizontal_scroll(page: Page) -> bool:
    return page.evaluate(
        "document.documentElement.scrollWidth <= document.documentElement.clientWidth"
    )


def test_research_first_browser_flow(browser_demo):
    local_url, root = browser_demo
    screenshot_dir = Path("test-output/ui")
    screenshot_dir.mkdir(parents=True, exist_ok=True)
    live_images = [
        _image(root / f"live-{index}.png", (25 * index, 80, 180 - 20 * index))
        for index in range(1, 6)
    ]

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page(viewport={"width": 1366, "height": 768})
        page.goto(local_url, wait_until="domcontentloaded")

        scope = page.locator("#scope-card")
        expect(scope).to_contain_text("một category sản phẩm công nghiệp")
        expect(scope).to_contain_text("Không hỗ trợ")
        page.screenshot(path=screenshot_dir / "default.png", full_page=True)
        assert _no_horizontal_scroll(page)

        page.get_by_role("tab", name="Kết quả chuẩn bị sẵn").click()
        expect(page.locator("#prepared-summary")).to_contain_text(
            "Prepared result - not live inference"
        )
        expect(page.get_by_text("Ground truth", exact=True)).to_be_visible()
        prepared_thumbnail = page.get_by_role("button", name="Thumbnail 1 of 1")
        prepared_thumbnail.scroll_into_view_if_needed()
        expect(prepared_thumbnail).to_be_visible()
        prepared_image = prepared_thumbnail.locator("img")
        prepared_image.evaluate("image => image.decode()")
        assert prepared_thumbnail.bounding_box()["height"] <= 340
        page.screenshot(path=screenshot_dir / "prepared-result.png")

        page.get_by_role("tab", name="Thử nghiệm tập ảnh mới").click()
        file_input = page.locator("#live-upload input[type=file]")
        file_input.set_input_files(str(live_images[0]))
        expect(page.locator("#validation-report")).to_contain_text(
            "Ảnh hợp lệ: 1", timeout=10_000
        )
        expect(page.locator("#run-live-inference")).to_be_disabled()
        expect(page.locator("#live-results")).not_to_be_visible()
        page.screenshot(path=screenshot_dir / "validation-error.png", full_page=True)

        page.get_by_role("button", name="Clear").click()
        file_input.set_input_files([str(path) for path in live_images])
        expect(page.locator("#validation-report")).to_contain_text(
            "Ảnh hợp lệ: 5", timeout=10_000
        )
        expect(page.locator("#experimental-warning")).to_contain_text("5-39")
        expect(page.locator("#run-live-inference")).to_be_disabled()

        page.locator("#input-attestation input[type=checkbox]").check()
        expect(page.locator("#run-live-inference")).to_be_disabled()
        page.locator("#initialize-model").click()
        expect(page.locator("#environment-card")).to_contain_text(
            "MuSc model: Sẵn sàng", timeout=10_000
        )
        expect(page.locator("#run-live-inference")).to_be_enabled(timeout=10_000)
        page.locator("#run-live-inference").click()
        page.wait_for_timeout(250)
        progress_bar = page.locator('[role="progressbar"]')
        expect(progress_bar).to_have_count(1)
        progress_bar.scroll_into_view_if_needed()
        page.screenshot(path=screenshot_dir / "live-progress.png")
        expect(page.locator("#live-result-summary")).to_contain_text(
            "Đã xếp hạng 5 ảnh", timeout=20_000
        )
        expect(page.locator("#live-results")).to_be_visible()
        page.screenshot(path=screenshot_dir / "live-result.png", full_page=True)

        assert _no_horizontal_scroll(page)
        expect(page.get_by_text("Heatmap", exact=True).last).to_be_visible()
        expect(page.get_by_text("Overlay", exact=True).last).to_be_visible()
        page.get_by_role("button", name="Thumbnail 2 of 5").click()
        expect(page.locator("#live-original a").first).to_have_attribute(
            "href", re.compile("live-4\\.png")
        )
        browser.close()
