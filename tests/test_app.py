from pathlib import Path
import warnings

import pytest

import musc_demo.app as app_module
from musc_demo.app import (
    GPU_CONCURRENCY_ID,
    GPU_CONCURRENCY_LIMIT,
    DemoServices,
    allocate_run_workspace,
    build_app,
    default_services,
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


def test_default_services_eagerly_reuses_cached_engine_when_cuda_is_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cached_engine = FakeEngine("ready")
    calls: list[tuple[Path, Path]] = []
    monkeypatch.setenv("MUSC_EAGER_INIT", "1")
    monkeypatch.delenv("MUSC_PREPARED_ROOT", raising=False)
    monkeypatch.setattr(app_module, "_detect_cuda", lambda: (True, "Fake T4"))
    monkeypatch.setattr(
        app_module,
        "get_cached_engine",
        lambda musc_root, config_path: (
            calls.append((musc_root, config_path)) or cached_engine
        ),
    )

    services = default_services()

    assert services.engine is cached_engine
    assert len(calls) == 1
    assert calls[0][1].name == "demo.json"


def test_default_services_preserves_prepared_fallback_when_eager_init_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fallback_engine = FakeEngine("uninitialized")
    monkeypatch.setenv("MUSC_EAGER_INIT", "1")
    monkeypatch.delenv("MUSC_PREPARED_ROOT", raising=False)
    monkeypatch.setattr(app_module, "_detect_cuda", lambda: (True, "Fake T4"))
    monkeypatch.setattr(
        app_module,
        "get_cached_engine",
        lambda *_: (_ for _ in ()).throw(RuntimeError("weights unavailable")),
    )
    monkeypatch.setattr(
        app_module,
        "MuScInferenceEngine",
        lambda *_: fallback_engine,
    )

    services = default_services()

    assert services.engine is fallback_engine
    assert services.engine.state == "uninitialized"
    assert services.prepared.error == "MUSC_PREPARED_ROOT is not configured"
