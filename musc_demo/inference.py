"""Ground-truth-free MuSc inference for one uploaded image category."""

from __future__ import annotations

import importlib
import json
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal, Protocol

import numpy as np

from .dataset_adapter import UnlabeledImageDataset
from .upstream import PINNED_MUSC_COMMIT, UpstreamModules, load_upstream_modules


ProgressCallback = Callable[[str, float, str], None]
EngineState = Literal["uninitialized", "ready", "failed"]


@dataclass(frozen=True)
class InferenceItem:
    image_id: str
    source_name: str
    path: Path
    raw_score: float
    adjusted_score: float
    anomaly_map: np.ndarray


@dataclass(frozen=True)
class InferenceResult:
    items: tuple[InferenceItem, ...]
    warnings: tuple[str, ...]
    config: dict[str, object]
    upstream_commit: str


@dataclass(frozen=True)
class FeatureBundle:
    class_tokens: np.ndarray
    patch_batches: tuple[tuple[object, ...], ...]


class InitializationError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


class InferenceStageError(RuntimeError):
    def __init__(self, stage: str, code: str, message: str) -> None:
        self.stage = stage
        self.code = code
        super().__init__(message)


class InferenceBackend(Protocol):
    def cuda_available(self) -> bool: ...

    def load_backbone(self, config: dict[str, object]) -> tuple[object, object]: ...

    def extract_features(
        self,
        dataset: UnlabeledImageDataset,
        model: object,
        config: dict[str, object],
        progress: ProgressCallback,
    ) -> object: ...

    def compute_anomaly_maps(
        self,
        features: object,
        config: dict[str, object],
        progress: ProgressCallback,
    ) -> np.ndarray: ...

    def adjust_scores(
        self,
        raw_scores: np.ndarray,
        class_tokens: np.ndarray,
        neighbors: tuple[int, ...],
    ) -> np.ndarray: ...

    def is_cuda_oom(self, error: BaseException) -> bool: ...

    def empty_cache(self) -> None: ...


class TorchMuScBackend:
    """Thin runtime adapter around the pinned official MuSc modules."""

    def __init__(self, musc_root: Path) -> None:
        self.musc_root = Path(musc_root)
        self.torch = importlib.import_module("torch")
        self.modules: UpstreamModules | None = None
        self.device: object | None = None

    def cuda_available(self) -> bool:
        return bool(self.torch.cuda.is_available())

    def load_backbone(self, config: dict[str, object]) -> tuple[object, object]:
        self.modules = load_upstream_modules(self.musc_root)
        self.device = self.torch.device("cuda:0")
        model, _, preprocess = self.modules.open_clip.create_model_and_transforms(
            str(config["backbone_name"]),
            int(config["image_size"]),
            pretrained=str(config["pretrained"]),
        )
        model.to(self.device)
        model.eval()
        return model, preprocess

    def extract_features(
        self,
        dataset: UnlabeledImageDataset,
        model: object,
        config: dict[str, object],
        progress: ProgressCallback,
    ) -> FeatureBundle:
        progress("normalize", 0.05, "Chuẩn hóa ảnh")
        batch_size = int(config["batch_size"])
        feature_layers = [int(layer) + 1 for layer in config["feature_layers"]]
        class_tokens: list[np.ndarray] = []
        patch_batches: list[tuple[object, ...]] = []
        progress("clip", 0.20, "Trích xuất CLIP features")

        for start in range(0, len(dataset), batch_size):
            items = [
                dataset[index]
                for index in range(start, min(start + batch_size, len(dataset)))
            ]
            batch = self.torch.stack([item["image"] for item in items]).to(
                dtype=self.torch.float32, device=self.device
            )
            with self.torch.no_grad(), self.torch.cuda.amp.autocast():
                image_features, patch_tokens = model.encode_image(batch, feature_layers)
                image_features = image_features / image_features.norm(
                    dim=-1, keepdim=True
                )
            class_tokens.extend(image_features.detach().cpu().numpy())
            patch_batches.append(
                tuple(token.detach().cpu() for token in patch_tokens)
            )

        return FeatureBundle(
            class_tokens=np.asarray(class_tokens),
            patch_batches=tuple(patch_batches),
        )

    def compute_anomaly_maps(
        self,
        features: FeatureBundle,
        config: dict[str, object],
        progress: ProgressCallback,
    ) -> np.ndarray:
        if self.modules is None or self.device is None:
            raise RuntimeError("MuSc backend is not initialized")
        if not features.patch_batches:
            raise ValueError("The image batch is empty")

        runtime_layers = [int(layer) + 1 for layer in config["feature_layers"]]
        feature_dim = int(features.patch_batches[0][0].shape[-1])
        degree_maps: list[object] = []
        progress("lnamd", 0.45, "Tính LNAMD")

        for degree_index, degree in enumerate(config["r_list"]):
            lnamd = self.modules.lnamd_factory(
                device=self.device,
                r=int(degree),
                feature_dim=feature_dim,
                feature_layer=runtime_layers,
            )
            layer_features: dict[int, list[object]] = {
                index: [] for index in range(len(runtime_layers))
            }
            for patch_batch in features.patch_batches:
                device_tokens = [token.to(self.device) for token in patch_batch]
                with self.torch.no_grad(), self.torch.cuda.amp.autocast():
                    embedded = lnamd._embed(device_tokens)
                    embedded = embedded / embedded.norm(dim=-1, keepdim=True)
                for layer_index in range(len(runtime_layers)):
                    layer_features[layer_index].append(
                        embedded[:, :, layer_index, :]
                    )

            if degree_index == 0:
                progress("msm", 0.75, "Tính mutual scoring")
            layer_maps: list[object] = []
            for layer_index in range(len(runtime_layers)):
                z = self.torch.cat(layer_features[layer_index], dim=0).to(self.device)
                layer_map = self.modules.msm(
                    Z=z,
                    device=self.device,
                    topmin_min=float(config["msm_interval"][0]),
                    topmin_max=float(config["msm_interval"][1]),
                )
                layer_maps.append(layer_map.cpu())
                self.torch.cuda.empty_cache()
            degree_maps.append(self.torch.stack(layer_maps).mean(dim=0))

        patch_maps = self.torch.stack(degree_maps).mean(dim=0).to(self.device)
        image_count, patch_count = patch_maps.shape
        side = math.isqrt(patch_count)
        if side * side != patch_count:
            raise ValueError(f"MuSc patch count {patch_count} is not square")
        functional = importlib.import_module("torch.nn.functional")
        resized = functional.interpolate(
            patch_maps.view(image_count, 1, side, side),
            size=int(config["image_size"]),
            mode="bilinear",
            align_corners=True,
        )
        result = resized.squeeze(1).detach().cpu().numpy()
        self.torch.cuda.empty_cache()
        return result

    def adjust_scores(
        self,
        raw_scores: np.ndarray,
        class_tokens: np.ndarray,
        neighbors: tuple[int, ...],
    ) -> np.ndarray:
        if self.modules is None:
            raise RuntimeError("MuSc backend is not initialized")
        return np.asarray(
            self.modules.rscin(raw_scores, class_tokens, k_list=list(neighbors)),
            dtype=np.float32,
        )

    def is_cuda_oom(self, error: BaseException) -> bool:
        oom_type = getattr(self.torch.cuda, "OutOfMemoryError", ())
        return isinstance(error, oom_type)

    def empty_cache(self) -> None:
        self.torch.cuda.empty_cache()


class MuScInferenceEngine:
    def __init__(
        self,
        musc_root: Path,
        config: Mapping[str, object],
        *,
        backend: InferenceBackend | None = None,
    ) -> None:
        self.musc_root = Path(musc_root)
        self.config = dict(config)
        self.backend = backend
        self._state: EngineState = "uninitialized"
        self._model: object | None = None
        self._preprocess: object | None = None

    @property
    def state(self) -> EngineState:
        return self._state

    @property
    def preprocess(self) -> object:
        if self._state != "ready" or self._preprocess is None:
            raise InitializationError("MODEL_NOT_READY", "MuSc model is not ready")
        return self._preprocess

    def initialize(self) -> None:
        if self._state == "ready":
            return
        try:
            if self.backend is None:
                self.backend = TorchMuScBackend(self.musc_root)
            if not self.backend.cuda_available():
                raise InitializationError(
                    "GPU_UNAVAILABLE", "Live inference requires a CUDA GPU"
                )
            self._model, self._preprocess = self.backend.load_backbone(self.config)
        except InitializationError:
            self._state = "failed"
            raise
        except Exception as error:
            self._state = "failed"
            raise InitializationError(
                "MODEL_INITIALIZATION_FAILED", str(error)
            ) from error
        self._state = "ready"

    def run(
        self,
        dataset: UnlabeledImageDataset,
        progress: ProgressCallback,
    ) -> InferenceResult:
        if self._state != "ready" or self._model is None:
            raise InitializationError("MODEL_NOT_READY", "MuSc model is not ready")
        if self.backend is None:
            raise InitializationError("MODEL_NOT_READY", "MuSc backend is not ready")

        current_stage = "normalize"

        def emit(stage: str, fraction: float, message: str) -> None:
            nonlocal current_stage
            current_stage = stage
            progress(stage, fraction, message)

        try:
            features = self.backend.extract_features(
                dataset, self._model, self.config, emit
            )
            maps = np.asarray(
                self.backend.compute_anomaly_maps(features, self.config, emit),
                dtype=np.float32,
            )
            if maps.ndim != 3 or maps.shape[0] != len(dataset.images):
                raise ValueError("MuSc returned an invalid anomaly-map batch")
            if not np.isfinite(maps).all():
                raise ValueError("MuSc returned non-finite anomaly maps")

            raw_scores = maps.reshape(maps.shape[0], -1).max(axis=1)
            emit("rscin", 0.90, "Điều chỉnh anomaly score bằng RsCIN")
            warnings: list[str] = []
            score_range = float(raw_scores.max() - raw_scores.min())
            if score_range <= np.finfo(np.float32).eps:
                adjusted_scores = raw_scores.copy()
                warnings.append("RSCIN_CONSTANT_SCORES")
            else:
                adjusted_scores = np.asarray(
                    self.backend.adjust_scores(
                        raw_scores,
                        np.asarray(features.class_tokens),
                        tuple(int(value) for value in self.config["rscin_neighbors"]),
                    ),
                    dtype=np.float32,
                )
            if adjusted_scores.shape != raw_scores.shape or not np.isfinite(
                adjusted_scores
            ).all():
                raise ValueError("RsCIN returned invalid adjusted scores")
        except Exception as error:
            if isinstance(error, (InitializationError, InferenceStageError)):
                raise
            if self.backend.is_cuda_oom(error):
                self.backend.empty_cache()
                raise InferenceStageError(
                    current_stage,
                    "CUDA_OOM",
                    "GPU memory is insufficient; reduce the image batch.",
                ) from error
            raise InferenceStageError(
                current_stage, "INFERENCE_FAILED", str(error)
            ) from error

        items = tuple(
            InferenceItem(
                image_id=image.image_id,
                source_name=image.source_name,
                path=image.path,
                raw_score=float(raw_scores[index]),
                adjusted_score=float(adjusted_scores[index]),
                anomaly_map=maps[index],
            )
            for index, image in enumerate(dataset.images)
        )
        return InferenceResult(
            items=items,
            warnings=tuple(warnings),
            config=dict(self.config),
            upstream_commit=str(
                self.config.get("upstream_commit", PINNED_MUSC_COMMIT)
            ),
        )


def _create_engine(
    musc_root: Path, config: dict[str, object]
) -> MuScInferenceEngine:
    return MuScInferenceEngine(musc_root, config)


@lru_cache(maxsize=1)
def get_cached_engine(
    musc_root: Path, config_path: Path
) -> MuScInferenceEngine:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    engine = _create_engine(Path(musc_root), config)
    engine.initialize()
    return engine

