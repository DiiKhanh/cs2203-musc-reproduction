from __future__ import annotations

import gc
import hashlib
import os
import sys
from pathlib import Path

import numpy as np
import pytest


pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_MUSC_GPU_TESTS") != "1",
    reason="set RUN_MUSC_GPU_TESTS=1 on a CUDA runtime",
)


def test_demo_engine_matches_upstream_on_mvtec_bottle(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    torch = pytest.importorskip("torch")
    musc_root = Path(os.environ["MUSC_ROOT"]).resolve()
    mvtec_root = Path(os.environ["MVTEC_ROOT"]).resolve()
    sys.path.insert(0, str(musc_root))

    from models.musc import MuSc

    import models.musc as upstream_musc

    captured: dict[str, np.ndarray] = {}

    def capture_metrics(
        gt_sp: np.ndarray,
        pr_sp: np.ndarray,
        gt_px: np.ndarray,
        pr_px: np.ndarray,
    ) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
        captured["adjusted"] = np.asarray(pr_sp).copy()
        captured["maps"] = np.asarray(pr_px).copy()
        return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0)

    monkeypatch.setattr(upstream_musc, "compute_metrics", capture_metrics)
    upstream_config = {
        "datasets": {
            "dataset_name": "mvtec_ad",
            "data_path": str(mvtec_root),
            "class_name": "bottle",
            "img_resize": 518,
            "divide_num": 1,
        },
        "models": {
            "backbone_name": "ViT-L-14-336",
            "pretrained": "openai",
            "batch_size": 4,
            "feature_layers": [5, 11, 17, 23],
            "r_list": [1, 3, 5],
        },
        "device": "0",
        "testing": {
            "output_dir": str(tmp_path / "upstream-output"),
            "vis": False,
            "vis_type": "whole_norm",
            "save_excel": False,
        },
    }
    upstream_engine = MuSc(upstream_config, seed=42)
    upstream_dataset = upstream_engine.load_datasets("bottle")
    upstream_paths = [
        Path(record[2]).resolve() for record in upstream_dataset.data_to_iterate
    ]
    upstream_engine.make_category_data("bottle")
    upstream_maps = captured["maps"].squeeze(1)
    upstream_raw = upstream_maps.reshape(len(upstream_maps), -1).max(axis=1)
    upstream_adjusted = captured["adjusted"]

    del upstream_dataset, upstream_engine
    gc.collect()
    torch.cuda.empty_cache()

    from musc_demo.dataset_adapter import UnlabeledImageDataset
    from musc_demo.inference import MuScInferenceEngine
    from musc_demo.models import ValidatedImage

    demo_config = {
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
    demo_engine = MuScInferenceEngine(musc_root, demo_config)
    demo_engine.initialize()
    validated_images = tuple(
        ValidatedImage(
            image_id=f"bottle-{index:03d}",
            source_name=str(path.relative_to(mvtec_root)),
            path=path,
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        )
        for index, path in enumerate(upstream_paths)
    )
    demo_dataset = UnlabeledImageDataset(
        validated_images, transform=demo_engine.preprocess
    )
    demo_result = demo_engine.run(demo_dataset, progress=lambda *_: None)
    demo_paths = [item.path.resolve() for item in demo_result.items]
    demo_maps = np.stack([item.anomaly_map for item in demo_result.items])
    demo_raw = np.asarray([item.raw_score for item in demo_result.items])
    demo_adjusted = np.asarray(
        [item.adjusted_score for item in demo_result.items]
    )

    assert demo_paths == upstream_paths
    assert np.max(np.abs(demo_raw - upstream_raw)) <= 1e-6
    assert np.max(np.abs(demo_adjusted - upstream_adjusted)) <= 1e-6
    assert np.max(np.abs(demo_maps - upstream_maps)) <= 1e-5
