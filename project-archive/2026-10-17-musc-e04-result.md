# MuSc E04 MVTec AD Top-1-Percent Pooling Result

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: validate
- Origin Date: 2026-10-17
- Verification Status: ANALYZED
- Version Label: validation_v1

## Run Identity

| Field | Value |
|---|---|
| Experiment | E04 top-1-percent mean pooling |
| Dataset | MVTec AD, all 15 categories |
| MuSc commit | `b76b93da8bd3096a99964a96ae29d46f197a0651` |
| Run ID | `20261016_102025_E04_mvtec_top1_pooling` |
| Run directory | `MyDrive/CS2203_MuSc/runs/20261016_102025_E04_mvtec_top1_pooling/` |
| Reviewed archive | [E04 quantitative ZIP](artifacts/e04/20261016_102025_E04_mvtec_top1_pooling.zip) |
| ZIP SHA-256 | `f86c6b7a10107c1b3aff6db9000e4d6b9f108504f90623c5dc7dad0510536a63` |
| Quantitative outcome | Accepted as complete; the preregistered hypothesis was not supported |
| Qualitative status | Completed and accepted; see [E04Q result note](2026-10-17-musc-e04q-result.md) |

## Artifact Integrity

The ZIP passed `testzip()` with no damaged member. The review also confirmed:

- All 15 categories have exactly one successful attempt.
- The archive contains 60 category-variant metric rows for four variants.
- All 1,725 per-image records have unique `(category, image_path)` keys.
- Every category JSON agrees exactly with the aggregate long-form metrics.
- Macro metrics recomputed from the 15 category rows agree exactly with the saved summary.
- The `bottle` gate passed and the seven E04 baseline means match E01 with a maximum delta of `0.0` percentage points.
- Every category used `top_count = 2684`, corresponding to `ceil(518 * 518 * 0.01)`.
- The interrupted first `transistor` attempt did not produce a success marker. The resumed second attempt completed and is the only successful `transistor` attempt.

The complete 15-category E04 run recorded 123.56 minutes of subprocess time across two Colab sessions.

## Frozen Comparison

E04 compared four image-level score variants computed from the same anomaly maps:

| Variant | Initial pooling | RsCIN |
|---|---|---|
| `max_raw` | Global maximum | No |
| `max_rscin` | Global maximum | Yes |
| `top1_raw` | Mean of the top 1 percent of pixels | No |
| `top1_rscin` | Mean of the top 1 percent of pixels | Yes |

The backbone, feature layers, input size, LNAMD degrees, MSM output, dataset split, and pixel-level anomaly maps were held fixed. E04 therefore supports conclusions about image-level aggregation only.

## Macro Results

| Variant | Image AUROC | Image F1-max | Image AP |
|---|---:|---:|---:|
| `max_raw` | 97.3864 | 96.7355 | 98.7258 |
| `max_rscin` | **97.7340** | **97.4057** | **99.0528** |
| `top1_raw` | 97.0227 | 96.8796 | 98.8246 |
| `top1_rscin` | 96.7811 | 96.9747 | 98.7766 |

The preregistered primary outcome was the macro Image AUROC difference between `top1_rscin` and `max_rscin`. It was `-0.9529` percentage points. The corresponding changes in Image F1-max and Image AP were `-0.4310` and `-0.2762` points.

Without RsCIN, top-1-percent pooling changed macro Image AUROC by `-0.3638`, Image F1-max by `+0.1441`, and Image AP by `+0.0988` points relative to maximum pooling.

## Paired Category Results

| Context | Metric | Wins | Ties | Losses | Mean delta | Median delta | Exact sign-test p |
|---|---|---:|---:|---:|---:|---:|---:|
| Without RsCIN | Image AUROC | 7 | 4 | 4 | -0.3638 | 0.0000 | 0.548828 |
| Without RsCIN | Image F1-max | 6 | 4 | 5 | +0.1441 | 0.0000 | 1.000000 |
| Without RsCIN | Image AP | 8 | 3 | 4 | +0.0988 | +0.0094 | 0.387695 |
| With RsCIN | Image AUROC | 5 | 4 | 6 | -0.9529 | 0.0000 | 1.000000 |
| With RsCIN | Image F1-max | 4 | 6 | 5 | -0.4310 | 0.0000 | 1.000000 |
| With RsCIN | Image AP | 5 | 4 | 6 | -0.2762 | 0.0000 | 1.000000 |

These sign tests are exploratory and remove tied categories. None is close to an uncorrected `0.05` threshold, so a multiple-comparison correction would not change the interpretation.

The largest category changes with RsCIN were:

- `screw`: Image AUROC `-9.7151`, Image F1-max `-3.2774`, and Image AP `-2.4585` points.
- `capsule`: Image AUROC `-3.6298` points.
- `metal_nut`: Image AUROC `+1.0264` and Image AP `+0.2397` points.
- `carpet`: Image AUROC `+0.1204`, Image F1-max `+0.5587`, and Image AP `+0.0379` points.

## Operating-Point Candidate Analysis

At the separately optimized F1-max operating point for each variant:

| Context | Maximum false positives resolved by top 1 percent | New false positives under top 1 percent |
|---|---:|---:|
| Without RsCIN | 17 | 25 |
| With RsCIN | 4 | 36 |

The candidate counts show a real trade-off. Top-1-percent pooling suppresses some high-scoring normal images, but it also changes score ordering in the opposite direction for more normal images under these post hoc thresholds. These counts are descriptive and must not be interpreted as deployment error rates.

## Fixed E04Q Case Selection

The qualitative follow-up uses nine cases selected from quantitative outputs before viewing E04Q heatmaps.

| Category | Relative test path | Quantitative role |
|---|---|---|
| `carpet` | `good/019.png` | False positive resolved in both raw and RsCIN contexts |
| `carpet` | `metal_contamination/013.png` | Largest RsCIN anomaly rank drop within `carpet` |
| `carpet` | `color/013.png` | Second-largest RsCIN anomaly rank drop within `carpet` |
| `transistor` | `good/022.png` | Largest rank drop among raw resolved false positives |
| `transistor` | `good/001.png` | New false-positive candidate under top 1 percent in both contexts |
| `transistor` | `cut_lead/001.png` | Largest RsCIN anomaly rank drop within `transistor` |
| `screw` | `good/008.png` | Largest adverse raw rank change among new false positives |
| `screw` | `good/003.png` | Largest adverse RsCIN rank change among new false positives |
| `screw` | `scratch_neck/000.png` | Largest RsCIN anomaly rank drop in the full dataset |

This set includes a positive case, mixed cases, and the strongest counterexample. It is not selected by visual attractiveness.

## Statistical and Methodological Warnings

- **Overall confidence: CAUTION.** The quantitative run is internally consistent, and E04Q now provides a limited mechanism explanation. The selected cases do not establish external generalization.
- F1-max thresholds are optimized on benchmark labels separately for each variant. F1 and false-positive candidate counts are therefore post hoc operating-point descriptions.
- AP and F1 depend on category prevalence. AUROC, AP, F1, paired category deltas, and confusion counts should be read together.
- The 15 categories are the complete MVTec AD benchmark, not a random sample of industrial anomaly domains. Inferential p-values do not establish external generalization.
- E04 changed image-level scoring only. It did not change segmentation maps or pixel-level metrics.

## Fallacy Scan

Coverage: 11/11 statistical fallacy types checked.

| Fallacy | Assessment |
|---|---|
| Simpson's paradox | No hidden aggregate reversal is claimed; macro and category-level directions are both reported. Category heterogeneity remains material. |
| Ecological fallacy | No image-level behavior is inferred solely from category means. Per-image records are preserved. |
| Berkson's paradox | No category was filtered after observing its result; all 15 benchmark categories are included. |
| Collider bias | No additional covariate was conditioned on in the 2 x 2 comparison. |
| Base-rate neglect | CAUTION: F1 and AP are prevalence-sensitive, so they are reported beside AUROC and confusion counts. |
| Regression to the mean | Not applicable because E04 is not a pre-post design selected on an extreme baseline. |
| Survivorship bias | All 15 categories completed. The interrupted `transistor` attempt was retained and the resumed success was documented. |
| Look-elsewhere effect | CAUTION: six paired metric comparisons are exploratory. None approaches nominal significance. |
| Garden of forking paths | The fraction, primary outcome, variants, and category set were fixed before the run. E04Q cases are fixed before heatmap review. |
| Correlation versus causation | CAUTION: metric changes alone do not establish the isolated-hotspot mechanism. E04Q is required. |
| Reverse causality | Not applicable to this controlled scoring comparison. |

## Decision

E04 is accepted as a complete and reproducible quantitative extension. The specific hypothesis that top-1-percent mean pooling improves image-level scoring over maximum pooling is not supported on the full MVTec AD benchmark. The primary outcome is negative, most paired medians are zero, and the strongest effect is a large degradation on `screw`.

The completed [E04Q qualitative review](2026-10-17-musc-e04q-result.md) shows that top-1-percent pooling can suppress some maximum-driven normal responses, but the positive cases also contain broad structured activation rather than only one isolated noisy patch. The `screw` panels show strong object-wide activation on both normal and anomalous images, which helps explain the adverse score ordering and threshold transitions. E04Q therefore explains the trade-off without reversing the negative full-benchmark conclusion.
