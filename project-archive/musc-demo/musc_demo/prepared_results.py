"""Manifest-driven access to immutable prepared benchmark artifacts."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path


PREPARED_BADGE = "Prepared result - not live inference"


@dataclass(frozen=True)
class PreparedAssets:
    original: Path | None
    ground_truth: Path | None
    heatmap: Path | None
    overlay: Path | None


@dataclass(frozen=True)
class PreparedExample:
    run_id: str
    image_id: str
    source_name: str
    defect_type: str
    role: str
    raw_score: float | None
    adjusted_score: float | None
    assets: PreparedAssets
    notices: tuple[str, ...]
    badge: str = PREPARED_BADGE


@dataclass(frozen=True)
class PreparedRun:
    run_id: str
    experiment_id: str
    dataset: str
    category: str
    timestamp: str
    source_commit: str
    description: str
    sample_counts: dict[str, int]
    defect_types: tuple[str, ...]
    ground_truth_available: bool
    metrics: dict[str, float]
    examples: tuple[PreparedExample, ...]
    available: bool
    notices: tuple[str, ...]
    badge: str = PREPARED_BADGE


def _resolve_under(root: Path, relative: str) -> Path | None:
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        return None
    return candidate


def _optional_asset(
    artifact_root: Path,
    value: object,
    label: str,
    notices: list[str],
) -> Path | None:
    if value in (None, ""):
        return None
    if not isinstance(value, str):
        notices.append(f"Invalid optional asset path for {label}")
        return None
    path = _resolve_under(artifact_root, value)
    if path is None:
        notices.append(f"Unsafe optional asset path: {label}")
        return None
    if not path.is_file():
        notices.append(f"Missing optional asset: {label}")
        return None
    return path


def _finite_optional(value: object) -> float | None:
    if value is None:
        return None
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("score must be finite")
    return number


class PreparedCatalog:
    def __init__(
        self, runs: tuple[PreparedRun, ...] = (), error: str | None = None
    ) -> None:
        self._runs = runs
        self._by_id = {run.run_id: run for run in runs}
        self.error = error

    @classmethod
    def load(cls, manifest_path: Path, artifact_root: Path) -> "PreparedCatalog":
        artifact_root = Path(artifact_root).resolve()
        try:
            document = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
            if document.get("schema_version") != 1:
                raise ValueError("Unsupported prepared manifest schema")
            run_documents = document["runs"]
            if not isinstance(run_documents, list):
                raise ValueError("Prepared manifest runs must be a list")
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
            return cls(error=f"Prepared catalog unavailable: {error}")

        runs: list[PreparedRun] = []
        seen_ids: set[str] = set()
        for run_document in run_documents:
            try:
                run = cls._parse_run(run_document, artifact_root)
                if run.run_id in seen_ids:
                    raise ValueError(f"Duplicate run ID: {run.run_id}")
                seen_ids.add(run.run_id)
                runs.append(run)
            except (TypeError, ValueError, KeyError) as error:
                return cls(
                    tuple(runs),
                    error=f"Prepared catalog contains an invalid run: {error}",
                )
        return cls(tuple(runs))

    @staticmethod
    def _parse_run(document: dict[str, object], artifact_root: Path) -> PreparedRun:
        notices: list[str] = []
        available = True
        run_id = str(document["run_id"])

        required = document.get("required_artifacts", [])
        if not isinstance(required, list):
            raise ValueError(f"required_artifacts must be a list for {run_id}")
        for relative in required:
            if not isinstance(relative, str):
                available = False
                notices.append("Invalid required artifact path")
                continue
            path = _resolve_under(artifact_root, relative)
            if path is None:
                available = False
                notices.append(f"Unsafe required artifact path: {relative}")
            elif not path.is_file():
                available = False
                notices.append(f"Missing required artifact: {relative}")

        metric_document = document.get("metrics", {})
        if not isinstance(metric_document, dict):
            raise ValueError(f"metrics must be an object for {run_id}")
        metrics: dict[str, float] = {}
        for name, value in metric_document.items():
            number = float(value)
            if not math.isfinite(number):
                available = False
                notices.append(f"Non-finite metric: {name}")
                continue
            metrics[str(name)] = number

        example_documents = document.get("examples", [])
        if not isinstance(example_documents, list):
            raise ValueError(f"examples must be a list for {run_id}")
        examples = tuple(
            PreparedCatalog._parse_example(run_id, example, artifact_root)
            for example in example_documents
        )
        counts_document = document.get("sample_counts", {})
        if not isinstance(counts_document, dict):
            raise ValueError(f"sample_counts must be an object for {run_id}")

        return PreparedRun(
            run_id=run_id,
            experiment_id=str(document["experiment_id"]),
            dataset=str(document["dataset"]),
            category=str(document["category"]),
            timestamp=str(document["timestamp"]),
            source_commit=str(document["source_commit"]),
            description=str(document["description"]),
            sample_counts={str(key): int(value) for key, value in counts_document.items()},
            defect_types=tuple(str(value) for value in document.get("defect_types", [])),
            ground_truth_available=bool(
                document.get("ground_truth_available", False)
            ),
            metrics=metrics,
            examples=examples,
            available=available,
            notices=tuple(notices),
        )

    @staticmethod
    def _parse_example(
        run_id: str, document: dict[str, object], artifact_root: Path
    ) -> PreparedExample:
        notices: list[str] = []
        asset_document = document.get("assets", {})
        if not isinstance(asset_document, dict):
            raise ValueError(f"assets must be an object for {run_id}")
        assets = PreparedAssets(
            original=_optional_asset(
                artifact_root, asset_document.get("original"), "original", notices
            ),
            ground_truth=_optional_asset(
                artifact_root,
                asset_document.get("ground_truth"),
                "ground_truth",
                notices,
            ),
            heatmap=_optional_asset(
                artifact_root, asset_document.get("heatmap"), "heatmap", notices
            ),
            overlay=_optional_asset(
                artifact_root, asset_document.get("overlay"), "overlay", notices
            ),
        )
        return PreparedExample(
            run_id=run_id,
            image_id=str(document["image_id"]),
            source_name=str(document["source_name"]),
            defect_type=str(document.get("defect_type", "unknown")),
            role=str(document.get("role", "neutral")),
            raw_score=_finite_optional(document.get("raw_score")),
            adjusted_score=_finite_optional(document.get("adjusted_score")),
            assets=assets,
            notices=tuple(notices),
        )

    def datasets(self) -> tuple[str, ...]:
        return tuple(sorted({run.dataset for run in self._runs}))

    def categories(self, dataset: str) -> tuple[str, ...]:
        return tuple(
            sorted({run.category for run in self._runs if run.dataset == dataset})
        )

    def runs(self, dataset: str, category: str) -> tuple[str, ...]:
        return tuple(
            run.run_id
            for run in self._runs
            if run.dataset == dataset and run.category == category
        )

    def get_run(self, run_id: str) -> PreparedRun:
        return self._by_id[run_id]

    def get_example(self, run_id: str, image_id: str) -> PreparedExample:
        run = self.get_run(run_id)
        return next(example for example in run.examples if example.image_id == image_id)
