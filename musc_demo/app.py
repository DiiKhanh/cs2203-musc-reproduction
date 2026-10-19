"""Gradio entry point for the research-first MuSc classroom demo."""

from __future__ import annotations

import json
import logging
import os
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from uuid import uuid4

import gradio as gr

from .dataset_adapter import UnlabeledImageDataset
from .export import export_run
from .inference import (
    InferenceResult,
    InitializationError,
    MuScInferenceEngine,
    ProgressCallback,
)
from .models import BatchValidationReport
from .prepared_results import PreparedCatalog, PreparedExample, PreparedRun
from .presenters import (
    SCOPE_CARD_MARKDOWN,
    custom_results_markdown,
    environment_card,
    experimental_warning,
    pipeline_error_markdown,
    prepared_result_markdown,
    run_button_state,
    validation_markdown,
)
from .validation import validate_uploads
from .visualization import RenderedItem, render_result_images


GPU_CONCURRENCY_ID = "musc-gpu"
GPU_CONCURRENCY_LIMIT = 1
LOGGER = logging.getLogger(__name__)


class DemoEngine(Protocol):
    @property
    def state(self) -> str: ...

    @property
    def preprocess(self) -> Callable[[object], object]: ...

    def initialize(self) -> None: ...

    def run(
        self, dataset: UnlabeledImageDataset, progress: ProgressCallback
    ) -> InferenceResult: ...


@dataclass(frozen=True)
class DemoServices:
    engine: DemoEngine
    prepared: PreparedCatalog
    workspace_root: Path
    gpu_available: bool
    gpu_name: str | None


def allocate_run_workspace(workspace_root: Path) -> Path:
    """Create a unique, never-reused child directory for one live attempt."""

    workspace_root = Path(workspace_root)
    workspace_root.mkdir(parents=True, exist_ok=True)
    while True:
        candidate = workspace_root / f"run-{uuid4().hex}"
        try:
            candidate.mkdir()
        except FileExistsError:
            continue
        return candidate


def _first(values: tuple[str, ...]) -> str | None:
    return values[0] if values else None


def _prepared_selection(
    catalog: PreparedCatalog,
    dataset: str | None,
    category: str | None,
    run_id: str | None,
) -> tuple[PreparedRun | None, PreparedExample | None]:
    if not dataset or not category or not run_id:
        return None, None
    try:
        run = catalog.get_run(run_id)
    except KeyError:
        return None, None
    if run.dataset != dataset or run.category != category:
        return None, None
    example = run.examples[0] if run.examples else None
    return run, example


def _prepared_gallery(run: PreparedRun | None) -> list[tuple[str, str]]:
    if run is None:
        return []
    gallery: list[tuple[str, str]] = []
    for example in run.examples:
        preview = example.assets.overlay or example.assets.heatmap or example.assets.original
        if preview is None:
            continue
        score = (
            f"{example.adjusted_score:.4f}"
            if example.adjusted_score is not None
            else "n/a"
        )
        gallery.append((str(preview), f"{example.source_name} | score {score}"))
    return gallery


def _example_values(
    example: PreparedExample | None,
) -> tuple[str | None, str | None, str | None, str | None, str]:
    if example is None:
        return None, None, None, None, "Chưa có ví dụ khả dụng."
    assets = example.assets
    details = (
        f"**Ảnh:** `{example.source_name}`  \n"
        f"**Vai trò minh họa:** {example.role}  \n"
        f"**Loại lỗi trong benchmark:** {example.defect_type}  \n"
        f"**Raw score / adjusted score:** {example.raw_score} / {example.adjusted_score}"
    )
    if example.notices:
        details += "\n\n" + "\n".join(f"- {value}" for value in example.notices)
    return (
        str(assets.original) if assets.original else None,
        str(assets.ground_truth) if assets.ground_truth else None,
        str(assets.heatmap) if assets.heatmap else None,
        str(assets.overlay) if assets.overlay else None,
        details,
    )


def theme_css_path() -> Path:
    return Path(__file__).with_name("theme.css")


def build_app(services: DemoServices) -> gr.Blocks:
    """Build the complete app without initializing or downloading the model."""

    datasets = services.prepared.datasets()
    initial_dataset = _first(datasets)
    categories = (
        services.prepared.categories(initial_dataset) if initial_dataset else ()
    )
    initial_category = _first(categories)
    runs = (
        services.prepared.runs(initial_dataset, initial_category)
        if initial_dataset and initial_category
        else ()
    )
    initial_run_id = _first(runs)
    initial_run, initial_example = _prepared_selection(
        services.prepared, initial_dataset, initial_category, initial_run_id
    )
    initial_environment = environment_card(
        gpu_available=services.gpu_available,
        gpu_name=services.gpu_name,
        engine_state=services.engine.state,
        prepared=services.prepared,
    )

    with gr.Blocks(
        title="MuSc - Industrial Anomaly Detection Demo",
        fill_width=True,
    ) as app:
        gr.Markdown(
            "# MuSc - Demo phát hiện bất thường công nghiệp",
            elem_id="demo-title",
        )
        with gr.Row(elem_classes="top-grid"):
            gr.Markdown(
                SCOPE_CARD_MARKDOWN,
                elem_id="scope-card",
                elem_classes="research-card",
            )
            with gr.Column(elem_classes="status-column"):
                environment_output = gr.Markdown(
                    initial_environment.markdown,
                    elem_id="environment-card",
                    elem_classes="research-card",
                )
                initialize_button = gr.Button(
                    "Khởi tạo MuSc model",
                    variant="primary",
                    interactive=initial_environment.initialize_interactive,
                    elem_id="initialize-model",
                )
                initialization_message = gr.Markdown("")

        with gr.Tabs():
            with gr.Tab("Kết quả chuẩn bị sẵn", id="prepared"):
                gr.Markdown(
                    "Kết quả dưới đây được đọc từ artifact của một run đã hoàn tất. "
                    "App không chạy lại model trong tab này.",
                    elem_classes="prepared-badge",
                )
                with gr.Row():
                    prepared_dataset = gr.Dropdown(
                        choices=list(datasets),
                        value=initial_dataset,
                        label="Dataset",
                        interactive=bool(datasets),
                    )
                    prepared_category = gr.Dropdown(
                        choices=list(categories),
                        value=initial_category,
                        label="Category",
                        interactive=bool(categories),
                    )
                    prepared_run = gr.Dropdown(
                        choices=list(runs),
                        value=initial_run_id,
                        label="Experiment / Run ID",
                        interactive=bool(runs),
                    )
                prepared_summary = gr.Markdown(
                    prepared_result_markdown(initial_run)
                    if initial_run
                    else "Chưa có prepared artifact khả dụng.",
                    elem_id="prepared-summary",
                )
                prepared_gallery = gr.Gallery(
                    _prepared_gallery(initial_run),
                    label="Ví dụ trong cùng run",
                    columns=4,
                    height=260,
                    object_fit="contain",
                    fit_columns=False,
                    elem_classes="result-gallery",
                )
                prepared_example = gr.Dropdown(
                    choices=[example.image_id for example in initial_run.examples]
                    if initial_run
                    else [],
                    value=initial_example.image_id if initial_example else None,
                    label="Ví dụ chi tiết",
                    interactive=bool(initial_run and initial_run.examples),
                )
                initial_assets = _example_values(initial_example)
                with gr.Row(elem_classes="image-grid prepared-images"):
                    prepared_original = gr.Image(
                        initial_assets[0], label="Ảnh gốc", interactive=False
                    )
                    prepared_ground_truth = gr.Image(
                        initial_assets[1], label="Ground truth", interactive=False
                    )
                    prepared_heatmap = gr.Image(
                        initial_assets[2], label="Heatmap", interactive=False
                    )
                    prepared_overlay = gr.Image(
                        initial_assets[3], label="Overlay", interactive=False
                    )
                prepared_example_details = gr.Markdown(initial_assets[4])

            with gr.Tab("Thử nghiệm tập ảnh mới", id="live"):
                gr.Markdown(
                    "### Hợp đồng input\n"
                    "Tải **5-200 ảnh JPEG/PNG** hoặc một ZIP. Tất cả ảnh phải thuộc "
                    "cùng một category sản phẩm công nghiệp. App chỉ kiểm tra file và số "
                    "lượng, không tự xác minh category hay domain.",
                    elem_classes="input-contract",
                )
                upload = gr.File(
                    label="Tập ảnh cùng một category",
                    file_count="multiple",
                    file_types=[".jpg", ".jpeg", ".png", ".zip"],
                    type="filepath",
                    elem_id="live-upload",
                )
                confirmation = gr.Checkbox(
                    label=(
                        "Tôi xác nhận tập ảnh thuộc một category sản phẩm công nghiệp "
                        "và có điều kiện chụp tương đối nhất quán."
                    ),
                    value=False,
                    elem_id="input-attestation",
                )
                validation_output = gr.Markdown(
                    validation_markdown(None), elem_id="validation-report"
                )
                experimental_output = gr.Markdown(
                    "",
                    visible=False,
                    elem_id="experimental-warning",
                    elem_classes="danger-card",
                )
                run_reason = gr.Markdown(
                    run_button_state(
                        None,
                        confirmed=False,
                        engine_state=services.engine.state,
                    ).reason
                )
                run_button = gr.Button(
                    "Chạy MuSc live inference",
                    variant="primary",
                    interactive=False,
                    elem_id="run-live-inference",
                )
                progress_output = gr.Markdown(
                    "Sẵn sàng chạy pipeline.", elem_id="pipeline-progress"
                )
                with gr.Column(visible=False, elem_id="live-results") as live_results:
                    result_summary = gr.Markdown("", elem_id="live-result-summary")
                    score_table = gr.Dataframe(
                        headers=["Hạng", "Tên tệp", "Raw score", "Adjusted score"],
                        value=[],
                        interactive=False,
                        label="Xếp hạng bằng anomaly score tương đối",
                    )
                    live_gallery = gr.Gallery(
                        [],
                        label="Ảnh xếp theo mức độ đáng ngờ",
                        columns=4,
                        height=300,
                        object_fit="contain",
                        fit_columns=False,
                        elem_classes="result-gallery",
                    )
                    with gr.Row(elem_classes="image-grid live-images"):
                        live_original = gr.Image(
                            label="Ảnh gốc",
                            interactive=False,
                            elem_id="live-original",
                        )
                        live_heatmap = gr.Image(
                            label="Heatmap",
                            interactive=False,
                            elem_id="live-heatmap",
                        )
                        live_overlay = gr.Image(
                            label="Overlay",
                            interactive=False,
                            elem_id="live-overlay",
                        )
                    with gr.Row():
                        csv_download = gr.File(label="Tải CSV", interactive=False)
                        zip_download = gr.File(
                            label="Tải ZIP kết quả", interactive=False
                        )

        report_state = gr.State(None)
        workspace_state = gr.State(None)
        rendered_state = gr.State(())

        def initialize_model():
            message = "MuSc model đã sẵn sàng."
            try:
                services.engine.initialize()
            except InitializationError:
                LOGGER.exception("MuSc model initialization failed")
                message = (
                    "Không thể khởi tạo MuSc model. Prepared Benchmark vẫn sử dụng được."
                )
            except Exception:
                LOGGER.exception("Unexpected MuSc model initialization failure")
                message = (
                    "Khởi tạo model không hoàn tất. Prepared Benchmark vẫn sử dụng được."
                )
            state = environment_card(
                gpu_available=services.gpu_available,
                gpu_name=services.gpu_name,
                engine_state=services.engine.state,
                prepared=services.prepared,
            )
            return (
                state.markdown,
                message,
                gr.Button(interactive=state.initialize_interactive),
            )

        initialize_button.click(
            initialize_model,
            outputs=[environment_output, initialization_message, initialize_button],
            queue=False,
            api_visibility="private",
        )

        def select_dataset(dataset: str | None):
            values = services.prepared.categories(dataset) if dataset else ()
            value = _first(values)
            run_values = (
                services.prepared.runs(dataset, value) if dataset and value else ()
            )
            return (
                gr.Dropdown(choices=list(values), value=value),
                gr.Dropdown(choices=list(run_values), value=_first(run_values)),
            )

        prepared_dataset.change(
            select_dataset,
            inputs=prepared_dataset,
            outputs=[prepared_category, prepared_run],
            queue=False,
            api_visibility="private",
        )

        def select_category(dataset: str | None, category: str | None):
            values = (
                services.prepared.runs(dataset, category)
                if dataset and category
                else ()
            )
            return gr.Dropdown(choices=list(values), value=_first(values))

        prepared_category.change(
            select_category,
            inputs=[prepared_dataset, prepared_category],
            outputs=prepared_run,
            queue=False,
            api_visibility="private",
        )

        def select_run(
            dataset: str | None, category: str | None, run_id: str | None
        ):
            run, example = _prepared_selection(
                services.prepared, dataset, category, run_id
            )
            assets = _example_values(example)
            return (
                prepared_result_markdown(run)
                if run
                else "Prepared run không khả dụng.",
                _prepared_gallery(run),
                gr.Dropdown(
                    choices=[item.image_id for item in run.examples] if run else [],
                    value=example.image_id if example else None,
                ),
                *assets,
            )

        prepared_run.change(
            select_run,
            inputs=[prepared_dataset, prepared_category, prepared_run],
            outputs=[
                prepared_summary,
                prepared_gallery,
                prepared_example,
                prepared_original,
                prepared_ground_truth,
                prepared_heatmap,
                prepared_overlay,
                prepared_example_details,
            ],
            queue=False,
            api_visibility="private",
        )

        def select_example(run_id: str | None, image_id: str | None):
            if not run_id or not image_id:
                return _example_values(None)
            try:
                example = services.prepared.get_example(run_id, image_id)
            except (KeyError, StopIteration):
                example = None
            return _example_values(example)

        prepared_example.change(
            select_example,
            inputs=[prepared_run, prepared_example],
            outputs=[
                prepared_original,
                prepared_ground_truth,
                prepared_heatmap,
                prepared_overlay,
                prepared_example_details,
            ],
            queue=False,
            api_visibility="private",
        )

        def validate_batch(files: list[str] | None, confirmed: bool):
            workspace = allocate_run_workspace(services.workspace_root)
            report = validate_uploads(
                [Path(value) for value in (files or [])], workspace
            )
            warning = experimental_warning(report)
            button = run_button_state(
                report,
                confirmed=confirmed,
                engine_state=services.engine.state,
            )
            return (
                validation_markdown(report),
                warning.markdown,
                gr.Markdown(visible=warning.visible),
                gr.Button(interactive=button.interactive),
                button.reason,
                report,
                str(workspace),
                gr.Column(visible=False),
            )

        upload.change(
            validate_batch,
            inputs=[upload, confirmation],
            outputs=[
                validation_output,
                experimental_output,
                experimental_output,
                run_button,
                run_reason,
                report_state,
                workspace_state,
                live_results,
            ],
            queue=False,
            api_visibility="private",
        )

        def update_confirmation(
            confirmed: bool, report: BatchValidationReport | None
        ):
            state = run_button_state(
                report,
                confirmed=confirmed,
                engine_state=services.engine.state,
            )
            return gr.Button(interactive=state.interactive), state.reason

        confirmation.change(
            update_confirmation,
            inputs=[confirmation, report_state],
            outputs=[run_button, run_reason],
            queue=False,
            api_visibility="private",
        )

        def run_live(
            report: BatchValidationReport | None,
            workspace_value: str | None,
            progress: gr.Progress = gr.Progress(),
        ):
            if report is None or not report.can_run or workspace_value is None:
                message = "Tập ảnh chưa đạt điều kiện để chạy."
                return (
                    message,
                    gr.Column(visible=False),
                    "",
                    [],
                    [],
                    None,
                    None,
                    None,
                    None,
                    None,
                    (),
                )
            workspace = Path(workspace_value)

            def emit(stage: str, fraction: float, message: str) -> None:
                progress(fraction, desc=message)

            try:
                dataset = UnlabeledImageDataset(report.images, services.engine.preprocess)
                result = services.engine.run(dataset, emit)
                rendered = render_result_images(result, workspace / "rendered", emit)
                bundle = export_run(result, rendered, workspace / "export")
            except Exception as error:
                LOGGER.exception("MuSc live pipeline failed")
                return (
                    pipeline_error_markdown(error),
                    gr.Column(visible=False),
                    "",
                    [],
                    [],
                    None,
                    None,
                    None,
                    None,
                    None,
                    (),
                )

            ranked = sorted(
                rendered, key=lambda item: (-item.adjusted_score, item.image_id)
            )
            first_item = ranked[0] if ranked else None
            gallery = [
                (
                    str(item.overlay_path),
                    f"#{rank} {item.source_name} | {item.adjusted_score:.4f}",
                )
                for rank, item in enumerate(ranked, start=1)
            ]
            table = [
                [rank, item.source_name, item.raw_score, item.adjusted_score]
                for rank, item in enumerate(ranked, start=1)
            ]
            return (
                "Hoàn tất pipeline: CLIP, LNAMD, MSM, RsCIN, heatmap và overlay.",
                gr.Column(visible=True),
                custom_results_markdown(result),
                table,
                gallery,
                str(first_item.original_path) if first_item else None,
                str(first_item.heatmap_path) if first_item else None,
                str(first_item.overlay_path) if first_item else None,
                str(bundle.csv_path),
                str(bundle.zip_path),
                tuple(ranked),
            )

        run_button.click(
            run_live,
            inputs=[report_state, workspace_state],
            outputs=[
                progress_output,
                live_results,
                result_summary,
                score_table,
                live_gallery,
                live_original,
                live_heatmap,
                live_overlay,
                csv_download,
                zip_download,
                rendered_state,
            ],
            concurrency_limit=GPU_CONCURRENCY_LIMIT,
            concurrency_id=GPU_CONCURRENCY_ID,
            trigger_mode="once",
            show_progress_on=progress_output,
            api_visibility="private",
        )

        def select_live_example(
            rendered: tuple[RenderedItem, ...], event: gr.SelectData
        ):
            index = event.index
            if isinstance(index, tuple):
                index = index[0]
            if not isinstance(index, int) or not 0 <= index < len(rendered):
                return None, None, None
            item = rendered[index]
            return (
                str(item.original_path),
                str(item.heatmap_path),
                str(item.overlay_path),
            )

        live_gallery.select(
            select_live_example,
            inputs=rendered_state,
            outputs=[live_original, live_heatmap, live_overlay],
            queue=False,
            api_visibility="private",
        )

    return app.queue(default_concurrency_limit=4, api_open=False)


def _detect_cuda() -> tuple[bool, str | None]:
    try:
        import torch

        if not torch.cuda.is_available():
            return False, None
        return True, str(torch.cuda.get_device_name(0))
    except (ImportError, RuntimeError):
        return False, None


def default_services() -> DemoServices:
    repository_root = Path(__file__).resolve().parents[1]
    musc_root = Path(os.environ.get("MUSC_ROOT", repository_root / "MuSc"))
    config_path = repository_root / "configs" / "demo.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    engine = MuScInferenceEngine(musc_root, config)

    prepared_root_value = os.environ.get("MUSC_PREPARED_ROOT")
    if prepared_root_value:
        prepared_root = Path(prepared_root_value)
        manifest_path = prepared_root / "manifest.json"
        prepared = PreparedCatalog.load(manifest_path, prepared_root)
    else:
        prepared = PreparedCatalog(error="MUSC_PREPARED_ROOT is not configured")
    gpu_available, gpu_name = _detect_cuda()
    workspace_root = Path(tempfile.mkdtemp(prefix="musc-demo-runs-"))
    return DemoServices(
        engine=engine,
        prepared=prepared,
        workspace_root=workspace_root,
        gpu_available=gpu_available,
        gpu_name=gpu_name,
    )


def launch_app(*, share: bool = True) -> None:
    build_app(default_services()).launch(share=share, css_paths=theme_css_path())


if __name__ == "__main__":
    launch_app()
