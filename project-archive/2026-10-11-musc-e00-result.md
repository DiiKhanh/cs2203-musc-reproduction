# MuSc E00 Reproduction Result

## Run Identity

| Field | Value |
|---|---|
| Date | 2026-10-11 |
| Experiment | E00 paper-aligned smoke test |
| Dataset and category | MVTec AD `bottle`, complete 83-image test split |
| MuSc commit | `b76b93da8bd3096a99964a96ae29d46f197a0651` |
| Dataset source | Hugging Face community mirror `micguida1/mvtech_anomaly_detection` |
| Dataset revision | `eb692b36147a3b3f12c561b879a062104c1f20b8` |
| Run directory | `MyDrive/CS2203_MuSc/runs/20261011_161100_E00_mvtec_bottle_clip_l14_336_baseline/` |
| Reviewed ZIP | `/Users/duykhanh/Downloads/20261011_161100_E00_mvtec_bottle_clip_l14_336_baseline-20261011T163344Z-1-001.zip` |
| Outcome | Accepted baseline reproduction |

## Configuration

| Item | Value |
|---|---|
| Backbone | OpenAI CLIP `ViT-L-14-336` |
| Input resolution | `518 x 518` |
| Feature layers | `[5, 11, 17, 23]` in the configuration |
| LNAMD degrees | `[1, 3, 5]` |
| MSM interval | Lowest 30 percent |
| `divide_num` | `1` |
| Batch size | `4` |
| Visualization | Enabled with `whole_norm` |
| Excel output | Enabled |

## Environment

| Component | Observed value |
|---|---|
| GPU | Tesla T4 |
| Python | 3.13.15 |
| PyTorch | 2.11.0+cu130 |
| Torchvision | 0.26.0+cu130 |
| CUDA runtime | 13.0 |
| OpenCV | 4.14.0 |
| scikit-learn | 1.6.1 |
| scikit-image | 0.25.2 |
| timm | 0.9.12 |

This environment differs from the repository's documented Python 3.8, PyTorch 2.0.1, and CUDA 11.7 environment. The reproduced metrics nevertheless match the repository reference after rounding.

## Quantitative Results

All values are percentages. Delta is reproduced minus reference.

| Level | Metric | Official repository | Reproduced E00 | Delta |
|---|---|---:|---:|---:|
| Image | AUROC | 99.920000 | 99.920635 | +0.000635 |
| Image | F1-max | 99.210000 | 99.212598 | +0.002598 |
| Image | AP | 99.980000 | 99.975198 | -0.004802 |
| Pixel | AUROC | 98.480000 | 98.477734 | -0.002266 |
| Pixel | F1-max | 79.170000 | 79.173999 | +0.003999 |
| Pixel | AP | 83.040000 | 83.044980 | +0.004980 |
| Pixel | PRO | 96.100000 | 96.104510 | +0.004510 |

The largest absolute difference is 0.004980 percentage points. Every reproduced value equals the official repository value when rounded to two decimal places. E00 therefore satisfies the numerical acceptance criterion.

## Runtime Evidence

| Measurement | Observed value |
|---|---:|
| Notebook wall time for the run cell | 335.097 seconds |
| MuSc time reported by the implementation | 2015.311 ms per image |
| Feature extraction | 249.810 ms per image |
| LNAMD degree 1 | 21.888 ms per image |
| MSM degree 1 | 489.090 ms per image |
| LNAMD degree 3 | 34.683 ms per image |
| MSM degree 3 | 565.381 ms per image |
| LNAMD degree 5 | 71.144 ms per image |
| MSM degree 5 | 579.506 ms per image |

The notebook wall time includes the first download of approximately 934 MB of CLIP weights, metric computation, visualization, and Drive output. It should not be compared directly with the paper's steady-state per-image inference time.

## Qualitative Output Check

The ZIP contains 83 anomaly maps:

| Test type | Number of maps |
|---|---:|
| `good` | 20 |
| `broken_large` | 20 |
| `broken_small` | 22 |
| `contamination` | 21 |
| Total | 83 |

Four representative maps were inspected. The normal example had a diffuse low response, while examples from `broken_large`, `broken_small`, and `contamination` contained localized higher-response regions. This is a basic output-integrity check rather than a formal qualitative accuracy assessment because the exported files contain heatmaps without original-image overlays or ground-truth contours.

## Compatibility Change

The first attempt failed before inference because Python imported the installed Hugging Face `datasets` package instead of MuSc's local `datasets` directory. The successful attempt:

1. Added `datasets/__init__.py`.
2. Changed the entry point to prepend the MuSc repository root to `sys.path`.

This changes module resolution only. It does not change feature extraction, LNAMD, MSM, RsCIN, anomaly maps, or metrics. The successful numerical match provides additional evidence that the compatibility change did not alter the method.

The same run directory was reused after the initial failure, so the original failed `console.log` was overwritten. Future runs must use a new timestamped directory for every attempt.

## Decision

E00 is accepted. The next experiment is E01: run the same frozen baseline over all 15 MVTec AD categories. Do not introduce top-k mean pooling until the full baseline result table is complete and preserved.
