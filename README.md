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

## Reproduction scope

- MVTec AD: all 15 categories, with quantitative and qualitative analysis.
- VisA: all 12 categories, with quantitative and qualitative analysis.
- Image-level metrics: AUROC, AP, and F1-max.
- Pixel-level metrics: AUROC, AP, F1-max, and PRO.
- Runtime environment: Google Colab with an NVIDIA Tesla T4.

The required baseline experiments are complete. E03 and E04 are optional
follow-up experiments and are intentionally not included in this package yet.

## Repository layout

```text
configs/       Frozen baseline configuration
docs/          English Colab runbook and reproducibility notes
notebooks/     Executable E00 smoke-test notebook
patches/       Minimal changes applied to the pinned upstream revision
scripts/       Category-level execution helper
tools/         Notebook build utility
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

## Rebuilding the notebook

```bash
python tools/build_notebook.py \
  docs/E00_colab_runbook.md \
  notebooks/E00_mvtec_bottle_smoke_test.ipynb
```

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
and datasets retain their own licenses and terms.
