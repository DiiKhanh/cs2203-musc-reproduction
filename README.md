# MuSc Reproduction for CS2203

This repository contains the reproducibility package for the CS2203 course
project on MuSc, a training-free method for zero-shot industrial anomaly
classification and segmentation.

The package reproduces the official implementation at commit
`b76b93da8bd3096a99964a96ae29d46f197a0651`. It keeps the upstream algorithm
and baseline configuration fixed while documenting two Colab compatibility
changes: local package import precedence and early release of expired CPU/GPU
objects before metric allocation.

Package dependencies that are not fixed by the Colab runtime are pinned or
bounded in `requirements-colab.txt`. Every run records the versions actually
resolved by the runtime.

## Course deliverables

The Vietnamese report and presentation are archived alongside the code:

| Deliverable | Version | File |
|---|---|---|
| Final report, PDF | 2026-11-02 | [Read the report](reports/2026-11-02-musc-final-report-vi.pdf) |
| Final report, editable DOCX | 2026-11-02 | [Download the report source](reports/2026-11-02-musc-final-report-vi.docx) |
| Presentation slides, PDF | 2026-11-05 | [Read the slides](slides/2026-11-05-musc-reproduction-slides-vi.pdf) |

The presenter confirmed completion of the demo recording on 2026-11-13.

## Reproduction scope

- MVTec AD: all 15 categories, with quantitative and qualitative analysis.
- VisA: all 12 categories, with quantitative and qualitative analysis.
- Image-level metrics: AUROC, AP, and F1-max.
- Pixel-level metrics: AUROC, AP, F1-max, and PRO.
- Runtime environment: Google Colab with an NVIDIA Tesla T4.

The required baseline experiments are complete. The qualitative runs and the E04
pooling extension are covered by
[`docs/qualitative_and_e04_runbook.md`](docs/qualitative_and_e04_runbook.md). The
optional E03 experiment was not run.

## Repository layout

```text
configs/       Frozen baseline configuration and the E04Q fixed case plan
docs/          English runbooks and reproducibility notes
docs/figures/  Qualitative figures used in the report and presentation
notebooks/     Executable reproduction and classroom notebooks
patches/       Minimal changes applied to the pinned upstream revision
scripts/       Category-level execution helper
tests/         Tests for the demo and post-processing tools
tools/         Notebook build, case selection, summary, and figure utilities
musc_demo/     Native Gradio classroom demo
prepared/      Prepared-bundle schema and provenance guidance
reports/       Final Vietnamese report in PDF and editable DOCX formats
slides/        Vietnamese presentation PDF
```

Datasets, pretrained weights, raw experiment archives, and credentials are not
stored in this repository.

## Quick start in Google Colab

Open the E00 notebook directly in Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DiiKhanh/cs2203-musc-reproduction/blob/master/notebooks/E00_mvtec_bottle_smoke_test.ipynb)

The notebook runs the complete 83-image MVTec AD `bottle` test split and saves
the command, environment, console log, metrics, and anomaly maps to a new
timestamped directory in Google Drive.

For a text version of the same procedure, see
[`docs/E00_colab_runbook.md`](docs/E00_colab_runbook.md).

## Classroom demo

The native Gradio app makes the project boundary explicit before any live run.
Its custom input is one **unlabeled set of 5-200 JPEG/PNG images from one
industrial product category**, with 40-200 images recommended. The images
should have a reasonably consistent viewpoint, scale, background, and lighting.
The app validates file structure and count, but the presenter remains
responsible for confirming the category and industrial domain.

The custom-upload output is a relative anomaly score and ranking within that
uploaded set, plus a heatmap and overlay for each image. It is not a calibrated
defect label, a diagnosis, or a benchmark metric. The app has two independent
modes:

- **Prepared Benchmark** reads immutable artifacts from completed MVTec AD or
  VisA runs and can show their recorded metrics and qualitative examples.
- **Live Inference** runs the pinned MuSc pipeline on a newly uploaded,
  unlabeled image set when a compatible CUDA GPU and model weights are
  available.

Open the classroom launcher in Colab:

[![Open MuSc Classroom Demo in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DiiKhanh/cs2203-musc-reproduction/blob/master/notebooks/MuSc_classroom_demo.ipynb)

A Vietnamese setup version is available as
[`notebooks/MuSc_classroom_demo_vi.ipynb`](notebooks/MuSc_classroom_demo_vi.ipynb),
with its editable source in
[`docs/musc_demo_colab_runbook_vi.md`](docs/musc_demo_colab_runbook_vi.md).

The private repository requires a read-only `GITHUB_TOKEN` stored in Colab
Secrets. Prepared Benchmark additionally requires an external bundle whose root
contains `manifest.json`; set `MUSC_PREPARED_ROOT` to that mounted directory.
The expected bundle structure is documented in
[`prepared/README.md`](prepared/README.md). Full setup, commit verification,
fallback behavior, and launch instructions are in
[`docs/musc_demo_colab_runbook.md`](docs/musc_demo_colab_runbook.md).

Before classroom use, complete the remaining Tesla T4 and external-artifact
gates in [`docs/musc_demo_verification.md`](docs/musc_demo_verification.md).
The presentation sequence, scope statement, new-batch decision rule, and
fallback actions are in
[`docs/musc_demo_presentation_checklist.md`](docs/musc_demo_presentation_checklist.md).

For a local prepared-only launch, install `requirements-demo.txt`, optionally
set `MUSC_PREPARED_ROOT`, and run:

```bash
python -m musc_demo.app
```

Live Inference is scoped to industrial visual anomaly detection. It has not
been validated for medical or dental images, must not be presented as a medical
model, and must not be used for clinical decisions. A medical-domain study
would require a suitable dataset, domain-specific protocol, labels, metrics,
baselines, and validation beyond this course demo.

## Frozen baseline

| Setting | Value |
|---|---|
| Upstream repository | `https://github.com/xrli-U/MuSc.git` |
| Upstream commit | `b76b93da8bd3096a99964a96ae29d46f197a0651` |
| Backbone | OpenAI CLIP `ViT-L-14-336` |
| Input size | `518 x 518` |
| Feature layers | `5 11 17 23` |
| LNAMD degrees | `1 3 5` |
| MSM interval | Lowest 30 percent, fixed upstream |
| `divide_num` | `1` |
| Batch size | `4` |
| Seed | `42`, fixed in the upstream entry point |

The machine-readable copy is in [`configs/baseline.json`](configs/baseline.json).

## Manual setup

```bash
git clone https://github.com/DiiKhanh/cs2203-musc-reproduction.git
cd cs2203-musc-reproduction
git clone https://github.com/xrli-U/MuSc.git MuSc
git -C MuSc checkout --detach b76b93da8bd3096a99964a96ae29d46f197a0651
git -C MuSc apply "$PWD/patches/colab-import-compatibility.patch"
python3 -m pip install -r requirements-colab.txt
```

Do not install an old PyTorch wheel over the Colab runtime. The experiments
retained Colab's existing PyTorch and recorded the resolved versions in each run.

Run one category with:

```bash
./scripts/run_category.sh \
  "$PWD/MuSc" \
  /absolute/path/to/mvtec_anomaly_detection \
  mvtec_ad \
  bottle \
  /absolute/path/to/output
```

Set the final optional argument to `True` to save visualizations:

```bash
./scripts/run_category.sh "$PWD/MuSc" /absolute/path/to/data mvtec_ad bottle /absolute/path/to/output True
```

## Dataset responsibility

Download MVTec AD from its official dataset page and VisA from its official
release location. Follow each dataset's license and access terms. MuSc compares
every test image with other unlabeled test images from the same category, so do
not replace the complete category test split with a hand-selected subset.

## Compatibility patches

`colab-import-compatibility.patch` ensures that Python imports MuSc's local
`datasets` package instead of the Hugging Face package that may already be
installed in Colab. It changes import resolution only.

`cpu-memory.patch` releases expired feature tensors before allocating flattened
metric arrays and removes two unnecessary NumPy copies. It does not change the
feature extractor, LNAMD, MSM, RsCIN, anomaly maps, or metric formulas. The
patch was needed after Colab exhausted CPU RAM during the full MVTec AD run:

```bash
git -C MuSc apply "$PWD/patches/cpu-memory.patch"
```

The patch was reviewed as a calculation-preserving change, but a matched
same-category A/B run with and without the patch was not performed. This is a
documented reproducibility limitation rather than an experimentally proven
equivalence claim.

`e02q-per-image-scores.patch` and `e04-top1-pooling.patch` are the archived
experiment diffs. Each one already includes the memory cleanup, so apply them
instead of `cpu-memory.patch`, as described in the
[runbook](docs/qualitative_and_e04_runbook.md).

## Rebuilding the notebook

```bash
python tools/build_notebook.py \
  docs/E00_colab_runbook.md \
  notebooks/E00_mvtec_bottle_smoke_test.ipynb
```

## Qualitative figures

Each file in `docs/figures/` shows the original image, the ground truth, the
MuSc heatmap, and the overlay with the ground-truth contour. VisA masks store
defect class IDs from 1 to 7 rather than 255, so the ground truth is rendered as
a binary mask (`mask > 0`). A raw 0-255 display of a VisA mask looks almost black.

The figures are rendered from the E01Q and E02Q selected assets, which are not
stored here. Follow the [runbook](docs/qualitative_and_e04_runbook.md) to create
them, or extract the archived runs into `e01q/` and `e02q/` under one directory,
then run:

```bash
python3 -m pip install -r requirements-tools.txt
python3 tools/render_slide_strips.py \
  --source /absolute/path/to/runs \
  --output docs/figures
```

Run the tests for these tools with
`python3 -m pytest tests/test_qualitative_tools.py`.

## Primary references

- Xurui Li, Ziming Huang, Feng Xue, and Yu Zhou. "MuSc: Zero-Shot Industrial
  Anomaly Classification and Segmentation with Mutual Scoring of the Unlabeled
  Images." ICLR 2024. <https://openreview.net/forum?id=AHgc5SMdtd>
- Official MuSc implementation: <https://github.com/xrli-U/MuSc>
- MVTec AD: <https://www.mvtec.com/research-teaching/datasets/mvtec-ad>
- VisA paper: <https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/1653_ECCV_2022_paper.php>

## License and attribution

The original files in this repository are released under the [MIT License](LICENSE).
This repository does not redistribute the MuSc source code, datasets, or model
weights. The upstream MuSc implementation is also distributed under its own
[MIT License](https://github.com/xrli-U/MuSc/blob/b76b93da8bd3096a99964a96ae29d46f197a0651/LICENSE),
and datasets retain their own licenses and terms. Dataset-derived images in
`docs/figures/`, `reports/`, and `slides/` contain MVTec AD and VisA samples,
so the MIT License does not cover those images and they remain under their
dataset terms.
