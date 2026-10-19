from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import musc_demo.inference as inference_module
import musc_demo.upstream as upstream_module
from musc_demo.inference import (
    InferenceItem,
    InferenceResult,
    InferenceStageError,
    InitializationError,
    MuScInferenceEngine,
    get_cached_engine,
)
from musc_demo.models import ValidatedImage
from musc_demo.upstream import UpstreamVersionError, load_upstream_modules


class FakeCudaOom(RuntimeError):
    pass


class FakeBackend:
    def __init__(
        self,
        maps: np.ndarray | None = None,
        *,
        cuda_available: bool = True,
        fail_stage: str | None = None,
        oom: bool = False,
    ) -> None:
        self.maps = (
            np.asarray(maps, dtype=np.float32)
            if maps is not None
            else np.asarray(
                [
                    [[1.0, 2.0], [3.0, 4.0]],
                    [[5.0, 6.0], [7.0, 8.0]],
                ],
                dtype=np.float32,
            )
        )
        self.has_cuda = cuda_available
        self.fail_stage = fail_stage
        self.oom = oom
        self.load_count = 0
        self.adjust_k: tuple[int, ...] | None = None
        self.adjust_called = False
        self.cache_cleared = False

    def cuda_available(self) -> bool:
        return self.has_cuda

    def load_backbone(self, config: dict[str, object]) -> tuple[object, object]:
        self.load_count += 1
        return object(), lambda image: image

    def extract_features(
        self, dataset: object, model: object, config: dict[str, object], progress: object
    ) -> object:
        progress("normalize", 0.05, "normalize")
        if self.fail_stage == "normalize":
            raise ValueError("normalization failed")
        progress("clip", 0.20, "clip")
        if self.fail_stage == "clip":
            raise ValueError("clip failed")
        return SimpleNamespace(class_tokens=np.eye(len(dataset.images), dtype=np.float32))

    def compute_anomaly_maps(
        self, features: object, config: dict[str, object], progress: object
    ) -> np.ndarray:
        progress("lnamd", 0.45, "lnamd")
        if self.fail_stage == "lnamd":
            raise ValueError("lnamd failed")
        progress("msm", 0.75, "msm")
        if self.fail_stage == "msm":
            if self.oom:
                raise FakeCudaOom("out of memory")
            raise ValueError("msm failed")
        return self.maps.copy()

    def adjust_scores(
        self,
        raw_scores: np.ndarray,
        class_tokens: np.ndarray,
        neighbors: tuple[int, ...],
    ) -> np.ndarray:
        self.adjust_called = True
        self.adjust_k = neighbors
        return np.linspace(0.25, 0.75, len(raw_scores), dtype=np.float32)

    def is_cuda_oom(self, error: BaseException) -> bool:
        return isinstance(error, FakeCudaOom)

    def empty_cache(self) -> None:
        self.cache_cleared = True


class DummyDataset:
    def __init__(self, count: int = 2) -> None:
        self.images = tuple(
            ValidatedImage(
                image_id=f"image-{index}",
                source_name=f"nested/image-{index}.png",
                path=Path(f"/tmp/image-{index}.png"),
                sha256=f"hash-{index}",
            )
            for index in range(count)
        )


def demo_config() -> dict[str, object]:
    return {
        "upstream_commit": "b76b93da8bd3096a99964a96ae29d46f197a0651",
        "backbone_name": "ViT-L-14-336",
        "pretrained": "openai",
        "image_size": 518,
        "feature_layers": [5, 11, 17, 23],
        "r_list": [1, 3, 5],
        "msm_interval": [0.0, 0.3],
        "batch_size": 4,
        "rscin_neighbors": [1, 2, 3],
    }


def ready_engine(backend: FakeBackend) -> MuScInferenceEngine:
    engine = MuScInferenceEngine(Path("/tmp/MuSc"), demo_config(), backend=backend)
    engine.initialize()
    return engine


def test_run_emits_inference_stages_in_order() -> None:
    engine = ready_engine(FakeBackend())
    stages: list[str] = []

    engine.run(
        DummyDataset(),
        progress=lambda stage, fraction, message: stages.append(stage),
    )

    assert stages == ["normalize", "clip", "lnamd", "msm", "rscin"]


def test_run_returns_raw_adjusted_scores_and_maps_in_input_order() -> None:
    maps = np.asarray(
        [
            [[1.0, 4.0], [2.0, 3.0]],
            [[8.0, 5.0], [6.0, 7.0]],
        ],
        dtype=np.float32,
    )
    engine = ready_engine(FakeBackend(maps))

    result = engine.run(DummyDataset(), progress=lambda *_: None)

    assert [item.image_id for item in result.items] == ["image-0", "image-1"]
    assert [item.source_name for item in result.items] == [
        "nested/image-0.png",
        "nested/image-1.png",
    ]
    assert [item.raw_score for item in result.items] == [4.0, 8.0]
    assert [item.adjusted_score for item in result.items] == [0.25, 0.75]
    np.testing.assert_array_equal(result.items[0].anomaly_map, maps[0])
    np.testing.assert_array_equal(result.items[1].anomaly_map, maps[1])


def test_custom_category_uses_rscin_neighbors_one_two_three() -> None:
    backend = FakeBackend()
    engine = ready_engine(backend)

    engine.run(DummyDataset(), progress=lambda *_: None)

    assert backend.adjust_k == (1, 2, 3)


def test_constant_scores_preserve_adjusted_scores_and_emit_warning() -> None:
    backend = FakeBackend(np.ones((2, 2, 2), dtype=np.float32))
    engine = ready_engine(backend)

    result = engine.run(DummyDataset(), progress=lambda *_: None)

    assert [item.raw_score for item in result.items] == [1.0, 1.0]
    assert [item.adjusted_score for item in result.items] == [1.0, 1.0]
    assert result.warnings == ("RSCIN_CONSTANT_SCORES",)
    assert backend.adjust_called is False


def test_result_contains_no_ground_truth_labels_or_metrics() -> None:
    result_field_names = {field.name for field in fields(InferenceResult)}
    item_field_names = {field.name for field in fields(InferenceItem)}

    assert result_field_names == {
        "items",
        "warnings",
        "config",
        "upstream_commit",
    }
    assert item_field_names == {
        "image_id",
        "source_name",
        "path",
        "raw_score",
        "adjusted_score",
        "anomaly_map",
    }
    assert {"label", "ground_truth", "metrics"}.isdisjoint(
        result_field_names | item_field_names
    )


def test_cached_engine_initializes_the_backbone_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "demo.json"
    config_path.write_text("{}", encoding="utf-8")
    created: list[object] = []

    class CachedFakeEngine:
        def initialize(self) -> None:
            created.append(self)

    monkeypatch.setattr(
        inference_module,
        "_create_engine",
        lambda musc_root, config: CachedFakeEngine(),
    )
    get_cached_engine.cache_clear()

    first = get_cached_engine(tmp_path / "MuSc", config_path)
    second = get_cached_engine(tmp_path / "MuSc", config_path)

    assert first is second
    assert created == [first]
    get_cached_engine.cache_clear()


def test_default_backend_is_created_only_when_engine_initializes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created: list[Path] = []

    class DeferredBackend(FakeBackend):
        def __init__(self, musc_root: Path) -> None:
            created.append(musc_root)
            super().__init__()

    monkeypatch.setattr(inference_module, "TorchMuScBackend", DeferredBackend)

    engine = MuScInferenceEngine(Path("/tmp/MuSc"), demo_config())

    assert created == []
    assert engine.state == "uninitialized"

    engine.initialize()

    assert created == [Path("/tmp/MuSc")]
    assert engine.state == "ready"


def test_initialize_without_cuda_reports_gpu_unavailable() -> None:
    engine = MuScInferenceEngine(
        Path("/tmp/MuSc"), demo_config(), backend=FakeBackend(cuda_available=False)
    )

    with pytest.raises(InitializationError) as error:
        engine.initialize()

    assert error.value.code == "GPU_UNAVAILABLE"
    assert engine.state == "failed"


def test_cuda_oom_clears_cache_and_returns_recovery_code() -> None:
    backend = FakeBackend(fail_stage="msm", oom=True)
    engine = ready_engine(backend)

    with pytest.raises(InferenceStageError) as error:
        engine.run(DummyDataset(), progress=lambda *_: None)

    assert error.value.stage == "msm"
    assert error.value.code == "CUDA_OOM"
    assert backend.cache_cleared is True


def test_failure_identifies_the_current_pipeline_stage() -> None:
    engine = ready_engine(FakeBackend(fail_stage="lnamd"))

    with pytest.raises(InferenceStageError) as error:
        engine.run(DummyDataset(), progress=lambda *_: None)

    assert error.value.stage == "lnamd"
    assert error.value.code == "INFERENCE_FAILED"
    assert isinstance(error.value.__cause__, ValueError)


def test_load_upstream_modules_rejects_wrong_commit_before_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    imported: list[str] = []
    monkeypatch.setattr(
        upstream_module.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout="wrong-commit\n"),
    )
    monkeypatch.setattr(
        upstream_module.importlib,
        "import_module",
        lambda name: imported.append(name),
    )

    with pytest.raises(UpstreamVersionError) as error:
        load_upstream_modules(tmp_path)

    assert error.value.actual == "wrong-commit"
    assert imported == []
