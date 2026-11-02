# Qualitative and E04 Runbook

This runbook regenerates the results behind the qualitative figures and the
pooling extension in the report and presentation. Complete the manual setup in
the [README](../README.md) first, then install the post-processing dependencies:

```bash
python3 -m pip install -r requirements-tools.txt
```

| Run | Dataset and scope | MuSc patch | Output used by |
|---|---|---|---|
| E01Q | MVTec AD: `leather`, `transistor`, `screw` | `cpu-memory.patch` | Report Figures 1-4, slides 2, 12, 13 |
| E02Q | VisA: `pcb2`, `pcb3`, `pcb4`, `fryum` | `e02q-per-image-scores.patch` | Report Figures 5-8 and the RsCIN table, slides 12-14 |
| E04 | MVTec AD: all 15 categories | `e04-top1-pooling.patch` | Pooling tables, slide 15 |
| E04Q | MVTec AD: `carpet`, `transistor`, `screw` | `e04-top1-pooling.patch` | Report Figure 9, slide 15 |

## Prepare MuSc

Use a separate clone for each patch. Both experiment patches already contain the
CPU memory cleanup, so do not also apply `cpu-memory.patch`.

```bash
git -C MuSc apply --ignore-space-change "$PWD/patches/e02q-per-image-scores.patch"
```

`--ignore-space-change` is required because the pinned `models/musc.py` uses CRLF
line endings while the archived diffs use LF.

For VisA, create `meta.json` in the dataset root with the upstream
`datasets/visa_preprocess.py` before the first run.

## E02Q

```bash
for category in pcb2 pcb3 pcb4 fryum; do
  ./scripts/run_category.sh "$PWD/MuSc" "$VISA_ROOT" visa "$category" "$RUNS/e02q/$category" True
done

python3 tools/select_qualitative_cases.py --experiment e02q \
  --results-root "$RUNS/e02q" --data-root "$VISA_ROOT" \
  --output "$RUNS/e02q/selected-qualitative-cases.csv"
python3 tools/build_qualitative_assets.py \
  --cases "$RUNS/e02q/selected-qualitative-cases.csv" \
  --output "$RUNS/e02q/selected-assets"
```

The patch writes `<category>_per-image-scores.csv` after the anomaly maps and
RsCIN scores exist. It does not change the values passed to `compute_metrics`.
Selection takes two low-scoring anomalies, two high-scoring normal images, and
the best and worst anomalies by per-image pixel AUROC for each category.

## E01Q

Run with `cpu-memory.patch`, then select from the saved heatmaps:

```bash
for category in leather transistor screw; do
  ./scripts/run_category.sh "$PWD/MuSc" "$MVTEC_ROOT" mvtec_ad "$category" "$RUNS/e01q/$category" True
done

python3 tools/select_qualitative_cases.py --experiment e01q \
  --results-root "$RUNS/e01q" --data-root "$MVTEC_ROOT" \
  --output "$RUNS/e01q/selected-qualitative-cases.csv"
python3 tools/build_qualitative_assets.py \
  --cases "$RUNS/e01q/selected-qualitative-cases.csv" \
  --output "$RUNS/e01q/selected-assets"
```

E01Q per-image pixel AUROC is reconstructed from 8-bit heatmaps. It ranks cases
and does not replace the raw-score metrics reported by MuSc. MVTec AD heatmaps
are lossless PNG files; this reconstruction does not apply to the VisA JPEG maps.

## E04

```bash
for category in bottle cable capsule carpet grid hazelnut leather metal_nut \
                pill screw tile toothbrush transistor wood zipper; do
  ./scripts/run_category.sh "$PWD/MuSc" "$MVTEC_ROOT" mvtec_ad "$category" "$RUNS/e04/$category"
done

python3 tools/summarize_e04.py --results-root "$RUNS/e04" --output "$RUNS/e04/summary"
```

The patch computes four image-level scores from one set of anomaly maps:
maximum or top-1-percent mean pooling, each with and without RsCIN. The
preregistered outcome is the macro Image AUROC of `top1_rscin` minus `max_rscin`.
F1-max thresholds are selected post hoc on benchmark labels, so the operating
points describe the benchmark and are not deployment error rates.

## E04Q

Keep the E04 patch. The nine cases were fixed from E04 scores before any E04Q
heatmap was reviewed, and they are listed in `configs/e04q-fixed-cases.csv`.

```bash
for category in carpet transistor screw; do
  ./scripts/run_category.sh "$PWD/MuSc" "$MVTEC_ROOT" mvtec_ad "$category" "$RUNS/e04q/$category" True
done

python3 tools/select_qualitative_cases.py --experiment e04q \
  --results-root "$RUNS/e04q" --data-root "$MVTEC_ROOT" \
  --cases configs/e04q-fixed-cases.csv \
  --output "$RUNS/e04q/selected-qualitative-cases.csv"
python3 tools/build_qualitative_assets.py \
  --cases "$RUNS/e04q/selected-qualitative-cases.csv" \
  --output "$RUNS/e04q/selected-assets"
```

## Figures

```bash
python3 tools/render_slide_strips.py --source "$RUNS" --output docs/figures
```

## Verification status

The accepted results came from Google Colab notebooks that added Google Drive
storage and resume handling around these steps. Those wrappers are not part of
this package.

Checked offline against the archived run outputs:

- Both experiment patches are the archived diffs and apply to the pinned commit
  after the import compatibility patch.
- Case selection reproduces the archived E01Q and E02Q selections and their order.
- Asset building matches the 51 stored cases, within one grey level for overlays.
- `summarize_e04.py` reproduces the archived E04 tables.
- The figure commands, run on a fixture rebuilt from the archived selected assets,
  reproduce the files in `docs/figures/` byte for byte.

Not checked: the commands above were not rerun on a GPU with the full datasets.
