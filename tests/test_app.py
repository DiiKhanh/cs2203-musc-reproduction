from pathlib import Path
import warnings

from musc_demo.app import (
    GPU_CONCURRENCY_ID,
    GPU_CONCURRENCY_LIMIT,
    DemoServices,
    allocate_run_workspace,
    build_app,
)
from musc_demo.prepared_results import PreparedCatalog


class FakeEngine:
    def __init__(self, state: str = "ready") -> None:
        self._state = state

    @property
    def state(self) -> str:
        return self._state


def _services(tmp_path: Path, state: str = "ready") -> DemoServices:
    return DemoServices(
        engine=FakeEngine(state),  # type: ignore[arg-type]
        prepared=PreparedCatalog(),
        workspace_root=tmp_path,
        gpu_available=state == "ready",
        gpu_name="Fake T4" if state == "ready" else None,
    )


def test_build_app_contains_prepared_and_live_tabs(tmp_path):
    app = build_app(_services(tmp_path))
    config = app.get_config_file()
    labels = {
        component["props"].get("label")
        for component in config["components"]
        if "props" in component
    }

    assert "Kết quả chuẩn bị sẵn" in labels
    assert "Thử nghiệm tập ảnh mới" in labels


def test_build_app_uses_no_deprecated_gradio_constructor_parameters(tmp_path):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        build_app(_services(tmp_path))

    assert not [
        warning
        for warning in caught
        if "parameters have been moved from the Blocks constructor" in str(warning.message)
    ]


def test_each_live_run_uses_an_isolated_workspace(tmp_path):
    first = allocate_run_workspace(tmp_path)
    second = allocate_run_workspace(tmp_path)

    assert first != second
    assert first.parent == tmp_path
    assert second.parent == tmp_path
    assert first.is_dir()
    assert second.is_dir()


def test_gpu_queue_allows_only_one_inference_job(tmp_path):
    app = build_app(_services(tmp_path))
    inference_dependencies = [
        function
        for function in app.fns.values()
        if function.concurrency_id == GPU_CONCURRENCY_ID
    ]

    assert GPU_CONCURRENCY_LIMIT == 1
    assert len(inference_dependencies) == 1
    assert inference_dependencies[0].concurrency_limit == 1
