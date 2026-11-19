# MuSc Qualitative Figure Correction

This note corrects how E02Q figures were displayed and how two cases were described. It does not change any experiment result. The E01Q and E02Q result notes and their archived ZIP files stay as recorded, and this note takes precedence for the points below.

## Display defect

VisA masks store defect class IDs from 1 to 7, while MVTec AD masks store 255. Cell Q7 of the [E02Q guide](2026-10-14-huong-dan-colab-musc-e02q-visa.md) displayed the raw mask on a 0-255 scale, so the Ground truth column of all 16 E02Q anomalous panels looks almost black.

- Evidence: `ground_truth.png` holds 5 for `pcb4/Anomaly/032`, 2 for `fryum/Anomaly/010`, and 4 for `pcb3/Anomaly/043`. All E01Q and E04Q anomalous masks hold 255.
- Unaffected: scores, per-image pixel AUROC, category metrics, heatmaps, overlays, and GT contours. Contours and metric masks already used `mask > 0` or binary targets.
- Overlay check: rebuilding each overlay from the stored original and heatmap matches the stored overlay within one grey level outside the contour.

## Figure changes

[`report-tools/render_slide_strips.py`](report-tools/render_slide_strips.py) renders binary-mask strips from the archived assets into `assets/musc-report/*-slide.png`. The original panels are kept.

| Output | Change |
|---|---|
| Slides 2, 12, 13 | Use the strips, with per-image AUROC labelled and its source stated |
| Report Figures 6, 7, 8 | Use the strips |
| Report Figures 1 to 5 | Unchanged, because their masks were displayed correctly or are empty |

## Interpretation changes

These measurements come from the stored 8-bit heatmaps, not from raw anomaly maps, which the compact ZIP files do not contain.

| Case | Earlier wording | Corrected reading | Evidence |
|---|---|---|---|
| `fryum/Anomaly/010` | Weak thin defect within an object-wide response | The strongest response lies on the background to the left, over 100 pixels from the defect on the 518 x 518 image; the thin edge defect is weaker | Peak outside the mask; about 866 of 906 pixels at or above the hottest smoothed in-mask value lie in two background regions |
| `transistor/cut_lead/008` | Thin defect submerged in the surrounding response | The peak lies at the cut lead, but most of the mask, the missing-lead region, is not prominent | Peak inside the mask; the median mask pixel is near the 67th percentile of non-mask pixels |

That the missing-lead region resembles background is a visual observation, not a demonstrated cause.

## Related wording

Slide 15 now states that the three post-RsCIN deltas are means over 15 categories. For Image AUROC, the E04 category comparison is 5 wins, 4 ties, and 6 losses, with a median delta of 0.
