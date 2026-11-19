# MuSc E02 VisA Reproduction Result

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: validate
- Origin Date: 2026-10-14
- Verification Status: ANALYZED
- Version Label: validation_v1

## Run Identity

| Field | Value |
|---|---|
| Execution dates | 2026-10-12 to 2026-10-13 UTC |
| Experiment | E02 full VisA quantitative baseline |
| Dataset scope | All 12 VisA categories, 2,162 test images |
| MuSc commit | `b76b93da8bd3096a99964a96ae29d46f197a0651` |
| Dataset source | Official Amazon S3 `VisA_20220922.tar` |
| Dataset SHA-256 | `2eb8690c803ab37de0324772964100169ec8ba1fa3f7e94291c9ca673f40f362` |
| Run ID | `20261012_171400_E02_visa_all_clip_l14_336_baseline` |
| Run directory | `MyDrive/CS2203_MuSc/runs/20261012_171400_E02_visa_all_clip_l14_336_baseline/` |
| Reviewed ZIP | `/Users/duykhanh/Downloads/20261012_171400_E02_visa_all_clip_l14_336_baseline.zip` |
| ZIP SHA-256 | `f6f8b52db29487f4c573db971cfa08426a9a809e7f718b4546c3ddb93da4c7ec` |
| Outcome | Accepted with documented image-level deviations and an official-reference inconsistency |

## Artifact Integrity

The ZIP integrity test completed without an error. A structured audit found no internal inconsistencies. The artifact contains:

- One successful `metrics.json`, `SUCCESS.json`, console log, command, and result workbook for each of the 12 VisA categories.
- Metric values that agree across every successful console log, `metrics.json`, category workbook, aggregate CSV, and aggregate XLSX container.
- Correctly recomputed unweighted means, deltas, alert rows, maximum absolute delta, and total successful subprocess time.
- The dataset source URL, archive checksum, category test counts, frozen configuration, upstream commit, environment records, and source patches.
- Two execution-session records with byte-identical environment, compatibility-fix, and CPU-memory-patch files.

The 12 successful subprocesses consumed 11,696.378 seconds, or 194.94 minutes. This excludes environment setup, dataset preparation, the interrupted attempt, and final aggregation.

## Frozen Configuration

| Item | Value |
|---|---|
| Backbone | OpenAI CLIP `ViT-L-14-336` |
| Input resolution | `518 x 518` |
| Feature layers | `[5, 11, 17, 23]` |
| LNAMD degrees | `[1, 3, 5]` |
| `divide_num` | `1` |
| Batch size | `4` |
| Visualization | Disabled for the quantitative run |
| Excel output | Enabled |
| Upstream seed | `42` |

All 13 commands, including the interrupted and resumed `macaroni2` attempts, retain this configuration.

## Environment

Both recorded sessions used the same environment:

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
| ftfy | 6.1.2 |
| openpyxl | 3.1.2 |

This is the same observed software stack as E01 and E01Q. The environment record reports ftfy 6.1.2 although the E02 guide requests 6.1.3. The artifact cannot determine whether this reflects package metadata or dependency resolution, so the difference is preserved as a reproducibility note.

## Mean Quantitative Results

All values and deltas are percentage points. The official repository reference is published to two decimal places.

| Level | Metric | Repository reference | Reproduced E02 | Delta |
|---|---|---:|---:|---:|
| Pixel | AUROC | 98.71 | 98.713645 | +0.003645 |
| Pixel | F1-max | 48.90 | 48.900484 | +0.000484 |
| Pixel | AP | 45.38 | 45.383142 | +0.003142 |
| Pixel | PRO | 92.43 | 92.428779 | -0.001221 |
| Image | AUROC | 92.57 | 92.400077 | -0.169923 |
| Image | F1-max | 89.06 | 88.797846 | -0.262154 |
| Image | AP | 93.31 | 93.177198 | -0.132802 |

The four pixel-level means match the official repository within 0.004 percentage points. The image-level means are lower by 0.133 to 0.262 percentage points.

## Official Reference Inconsistency

The pinned MuSc README reports `fryum` Pixel AUROC as 99.43, but the 12 displayed category values then average to 98.8383 rather than the displayed mean of 98.71. Holding the other 11 displayed rows fixed, the displayed mean implies a `fryum` value of 97.89. E02 produced 97.9309, which is only 0.0409 percentage points above that implied value.

The duplicated 99.43 values in the `fryum` Image AP and Pixel AUROC cells, together with the inconsistent mean, strongly indicate a typographical error in the official per-category table. This is an inference from the published table, not a replacement of the source record. The reviewed ZIP correctly preserves the official printed value, the resulting -1.4991 delta, and the corresponding alert.

After isolating this inconsistent reference cell, every pixel-level category comparison is within 0.043 percentage points. The largest remaining absolute deltas are:

| Metric | Largest absolute delta | Category |
|---|---:|---|
| Pixel AUROC | 0.004977 | `pipe_fryum` |
| Pixel F1-max | 0.016861 | `capsules` |
| Pixel AP | 0.022851 | `capsules` |
| Pixel PRO | 0.042343 | `chewinggum` |

## Image-Level Per-Category Agreement

Twenty-four of the 36 image-level comparisons exceed the 0.10 percentage-point review threshold: 8 AUROC, 10 F1-max, and 6 AP comparisons. The deltas are:

| Category | AUROC delta | F1-max delta | AP delta |
|---|---:|---:|---:|
| `candle` | -0.020000 | -0.350909 | -0.091351 |
| `capsules` | -0.253333 | -0.291386 | -0.092879 |
| `cashew` | -0.260000 | -0.185385 | -0.108468 |
| `chewinggum` | +0.000000 | -0.490404 | +0.001213 |
| `fryum` | -0.060000 | -0.004103 | -0.026237 |
| `macaroni1` | +0.170000 | -0.001379 | -0.215462 |
| `macaroni2` | -0.040000 | +0.252766 | -0.009683 |
| `pcb1` | -0.190000 | -0.149474 | -0.177646 |
| `pcb2` | -1.190000 | -2.506154 | -0.886897 |
| `pcb3` | -0.549703 | -0.938692 | -0.200662 |
| `pcb4` | +0.173960 | +0.987551 | +0.173099 |
| `pipe_fryum` | +0.120000 | +0.511724 | +0.041355 |

The largest deviation is `pcb2` Image F1-max at -2.5062 percentage points. The near-exact pixel-level agreement indicates that feature extraction, LNAMD, MSM, anomaly-map construction, and pixel metrics were reproduced closely. The remaining deviations are localized to image-level scoring, which includes RsCIN. As in E01, environment-sensitive neighbor selection or ordering in RsCIN is a plausible explanation, but no official-environment rerun or controlled attribution experiment was performed. This explanation remains an inference.

## Interrupted Run and Recovery

The first `macaroni2` attempt ends abruptly during layer-2 mutual scoring at item 143 of 200. It contains a command, partial output directory, and console log, but no `SUCCESS.json`, `FAILED.json`, metrics, or result workbook. The artifact records no exception, so it cannot distinguish a Colab disconnection, runtime termination, or memory exhaustion.

The next session reused the immutable E02 Run ID, kept all completed category outputs, used byte-identical environment and patch records, and completed `macaroni2/attempt_02` plus the remaining PCB and pipe categories. The interrupted attempt remains preserved rather than overwritten.

## Statistical Validation

- Overall Confidence: CAUTION
- Statistical content: fixed-benchmark point metrics only; no p-values, confidence intervals, effect sizes, or hypothesis tests are present.
- Multiple-comparison correction: not applicable because the comparison table is a complete descriptive audit, not a significance-testing procedure.
- Reproducibility method: artifact audit against the pinned official repository table; no independent local rerun.
- Reproducibility verdict: CANNOT_VERIFY under the experiment-agent rerun protocol. The Colab result is internally consistent and suitable for the course baseline, but it was not rerun independently in a matched environment during this audit.

### Fallacy Scan

- Coverage: 11/11 statistical fallacy types checked.

| Fallacy | Severity | Finding |
|---|---|---|
| Simpson's paradox | NOTE | No association is estimated. Both category rows and the unweighted category mean are retained. |
| Ecological fallacy | NOTE | No category-level metric is used to infer individual-image behavior. |
| Berkson's paradox | NOTE | No correlation analysis or selected clinical sample is involved. |
| Collider bias | NOTE | No covariate adjustment is performed. |
| Base-rate neglect | CAUTION | AP and F1-max depend on anomaly prevalence and category composition. They should not be compared without the fixed benchmark context. |
| Regression to the mean | NOTE | There is no pre-post comparison or extreme-group intervention. |
| Survivorship bias | NOTE | The interrupted attempt is preserved and disclosed; all 12 categories eventually completed, so no category was dropped from the metric table. |
| Look-elsewhere effect | NOTE | All seven metrics for all 12 categories are retained, including unfavorable deviations. |
| Garden of forking paths | NOTE | The run was not preregistered, but the baseline configuration was frozen before execution and all attempts were preserved. |
| Correlation and causation | NOTE | No causal claim is made from the benchmark comparison. |
| Reverse causality | NOTE | No directional causal relationship is estimated. |

## Decision

E02 is accepted as the full VisA quantitative baseline with explicit caveats because:

- All 12 categories and 2,162 test images completed under the frozen configuration.
- The artifact is internally consistent across logs, JSON, CSV, and workbook outputs.
- All four pixel-level mean metrics are within 0.004 percentage points of the official repository.
- The only apparent large pixel-level category deviation is explained by an internally inconsistent official reference cell and is preserved without rewriting the source artifact.
- Image-level deviations are material but localized, fully enumerated, and do not invalidate the pixel-level reproduction.

The next experiment should be a VisA qualitative analysis, prioritizing `pcb2`, `pcb3`, and `pcb4` to investigate the largest image-level deviations, plus `fryum` to visually support the reference-table diagnosis. E03 and E04 should remain optional until the VisA qualitative evidence and final report structure are secure.
