# MuSc E00 Colab Runbook

## Goal

Run the paper-aligned MuSc baseline on the complete MVTec AD `bottle` test split in Google Colab. This run verifies the end-to-end pipeline before any experimental modification is introduced.

The run is complete only when it produces:

- Image-level AUROC, F1-max, and AP.
- Pixel-level AUROC, F1-max, AP, and PRO.
- An Excel result file.
- Saved anomaly-map visualizations.
- An environment log and a console log.
- A copy of every output in Google Drive.

## Frozen E00 Configuration

| Item | Value |
|---|---|
| Upstream repository | `https://github.com/xrli-U/MuSc.git` |
| Git commit | `b76b93da8bd3096a99964a96ae29d46f197a0651` |
| Dataset | MVTec AD |
| Category | `bottle` |
| Backbone | OpenAI CLIP `ViT-L-14-336` |
| Input resolution | `518` |
| Feature layers in the configuration | `5 11 17 23` |
| LNAMD degrees | `1 3 5` |
| MSM interval | Lowest 30 percent, fixed in the implementation |
| Dataset divisions | `1` |
| Batch size | `4` |
| Random seed | `42`, fixed in `examples/musc_main.py` |
| Visualization normalization | `whole_norm` |

`whole_norm` is selected for E00 because it preserves relative anomaly-map strength across images. The quantitative metrics use raw anomaly maps and are unaffected by the visualization normalization setting.

## Dataset Source Options

### Option A - Official MVTec download

Download MVTec AD from the [official MVTec AD page](https://www.mvtec.com/research-teaching/datasets/mvtec-ad), follow its license terms, and extract the dataset in Google Drive so that this path exists:

```text
MyDrive/datasets/mvtec_anomaly_detection/bottle/test
```

The `bottle` directory must include all three original directories:

```text
bottle/
  ground_truth/
  test/
  train/
```

Do not build E00 from a manually selected subset. MuSc uses all unlabeled test images in the category as mutual-scoring evidence.

### Option B - Hugging Face mirror

The community mirror [`micguida1/mvtech_anomaly_detection`](https://huggingface.co/datasets/micguida1/mvtech_anomaly_detection) stores the original `mvtec_anomaly_detection.tar.xz` archive and declares the original CC BY-NC-SA 4.0 license. It is not an official MVTec repository. Record the mirror repository and its exact revision, and continue citing MVTec as the dataset source.

The archive is approximately 5.26 GB. The optional download cell below caches it in Google Drive and extracts it into Colab's local storage. This avoids slow image-by-image access from Drive during MuSc inference.

## Colab Runtime

In Colab, select **Runtime > Change runtime type > T4 GPU** or another available NVIDIA GPU. Then run the cells below in order.

### Cell 1 - Mount Drive and define paths

```python
from google.colab import drive
drive.mount("/content/drive")

from datetime import datetime
from pathlib import Path
import os

UPSTREAM_COMMIT = "b76b93da8bd3096a99964a96ae29d46f197a0651"
DATA_ROOT = Path("/content/drive/MyDrive/datasets/mvtec_anomaly_detection")
DATA_SOURCE = "official_mvtec_manual_download"
PROJECT_ROOT = Path("/content/drive/MyDrive/CS2203_MuSc")
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S_E00_mvtec_bottle_clip_l14_336_baseline")
RUN_ROOT = PROJECT_ROOT / "runs" / RUN_ID
OUTPUT_ROOT = RUN_ROOT / "output"

RUN_ROOT.mkdir(parents=True, exist_ok=False)
os.environ["MUSC_COMMIT"] = UPSTREAM_COMMIT
os.environ["MUSC_DATA_ROOT"] = str(DATA_ROOT)
os.environ["MUSC_RUN_ROOT"] = str(RUN_ROOT)
os.environ["MUSC_OUTPUT_ROOT"] = str(OUTPUT_ROOT)

print("Run ID:", RUN_ID)
print("Dataset:", DATA_ROOT)
print("Persistent output:", RUN_ROOT)
```

Creating a new timestamped directory prevents later runs from overwriting E00.

### Cell 2 - Verify GPU and record the initial environment

```python
import platform
import subprocess
import sys
import torch

assert torch.cuda.is_available(), "GPU is not enabled. Change the Colab runtime to a GPU."

gpu_name = torch.cuda.get_device_name(0)
environment_text = "\n".join([
    f"python={sys.version}",
    f"platform={platform.platform()}",
    f"torch={torch.__version__}",
    f"cuda_runtime={torch.version.cuda}",
    f"gpu={gpu_name}",
    f"upstream_commit={UPSTREAM_COMMIT}",
])

print(environment_text)
(RUN_ROOT / "environment-before-install.txt").write_text(environment_text, encoding="utf-8")
subprocess.run(["nvidia-smi"], check=True)
```

### Optional Cell 2A - Download MVTec AD from Hugging Face

Skip this cell when using the dataset already extracted in Google Drive.

```python
import os
import subprocess
import sys
from pathlib import Path

subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "huggingface_hub>=0.20,<2"],
    check=True,
)

import huggingface_hub
from huggingface_hub import HfApi, hf_hub_download

HF_DATASET_REPO = "micguida1/mvtech_anomaly_detection"
HF_DATASET_REVISION = HfApi().dataset_info(HF_DATASET_REPO).sha
HF_CACHE = PROJECT_ROOT / "cache" / "huggingface"

archive_path = hf_hub_download(
    repo_id=HF_DATASET_REPO,
    repo_type="dataset",
    filename="mvtec_anomaly_detection.tar.xz",
    revision=HF_DATASET_REVISION,
    cache_dir=HF_CACHE,
)

extract_root = Path("/content/data")
extract_root.mkdir(parents=True, exist_ok=True)
subprocess.run(
    ["tar", "-xJf", archive_path, "-C", str(extract_root)],
    check=True,
)

expected_root = extract_root / "mvtec_anomaly_detection"
if expected_root.is_dir():
    DATA_ROOT = expected_root
else:
    candidates = [path.parent.parent for path in extract_root.rglob("bottle/test")]
    candidates = sorted(set(candidates))
    assert len(candidates) == 1, f"Could not identify one MVTec AD root: {candidates}"
    DATA_ROOT = candidates[0]

DATA_SOURCE = f"huggingface:{HF_DATASET_REPO}@{HF_DATASET_REVISION}"
os.environ["MUSC_DATA_ROOT"] = str(DATA_ROOT)

(RUN_ROOT / "dataset-source.txt").write_text(
    f"source={DATA_SOURCE}\n"
    "archive=mvtec_anomaly_detection.tar.xz\n"
    f"huggingface_hub={huggingface_hub.__version__}\n",
    encoding="utf-8",
)

print("Dataset source:", DATA_SOURCE)
print("Extracted dataset:", DATA_ROOT)
```

### Cell 3 - Verify the MVTec AD `bottle` structure

```python
required_paths = [
    DATA_ROOT / "bottle" / "ground_truth",
    DATA_ROOT / "bottle" / "test",
    DATA_ROOT / "bottle" / "train",
]

for path in required_paths:
    assert path.is_dir(), f"Missing required directory: {path}"

supported_suffixes = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
test_images = sorted(
    path
    for path in (DATA_ROOT / "bottle" / "test").glob("*/*")
    if path.is_file() and path.suffix.lower() in supported_suffixes
)
assert len(test_images) == 83, (
    f"Expected exactly 83 bottle test images, found {len(test_images)}."
)

missing_masks = []
for image_path in test_images:
    defect_type = image_path.parent.name
    if defect_type == "good":
        continue
    expected_mask = (
        DATA_ROOT
        / "bottle"
        / "ground_truth"
        / defect_type
        / f"{image_path.stem}_mask.png"
    )
    if not expected_mask.is_file():
        missing_masks.append((image_path, expected_mask))

assert not missing_masks, f"Missing anomaly masks: {missing_masks[:5]}"
print(f"Verified {len(test_images)} bottle test images and their anomaly masks.")
```

The official MVTec AD `bottle` test split should contain 83 images. Stop and inspect the extraction if the count differs.

### Cell 4 - Clone and freeze the official MuSc revision

```bash
%%bash
set -euo pipefail
rm -rf /content/MuSc
git clone https://github.com/xrli-U/MuSc.git /content/MuSc
cd /content/MuSc
git checkout --detach "$MUSC_COMMIT"
git rev-parse HEAD | tee "$MUSC_RUN_ROOT/upstream-commit.txt"
```

The temporary `/content/MuSc` directory is recreated whenever the runtime restarts. The commit hash and experiment outputs remain in Drive.

### Cell 5 - Install dependencies without replacing Colab's PyTorch

```python
import subprocess
import sys

packages = [
    "ftfy==6.1.3",
    "huggingface_hub>=0.20,<2",
    "openpyxl==3.1.2",
    "timm==0.9.12",
]
subprocess.run([sys.executable, "-m", "pip", "install", "-q", *packages], check=True)
```

The official repository documents PyTorch 2.0.1, torchvision 0.15.2, and CUDA 11.7. Current Colab runtimes may use a newer compatible stack. E00 keeps Colab's existing PyTorch installation and records the actual versions instead of forcing an old CUDA wheel into the runtime.

### Cell 6 - Record all tested package versions

```python
import cv2
import ftfy
import huggingface_hub
import openpyxl
import platform
import sklearn
import skimage
import sys
import timm
import torch
import torchvision

versions = {
    "python": sys.version,
    "platform": platform.platform(),
    "torch": torch.__version__,
    "torchvision": torchvision.__version__,
    "cuda_runtime": torch.version.cuda,
    "gpu": torch.cuda.get_device_name(0),
    "opencv": cv2.__version__,
    "ftfy": ftfy.__version__,
    "huggingface_hub": huggingface_hub.__version__,
    "openpyxl": openpyxl.__version__,
    "scikit_learn": sklearn.__version__,
    "scikit_image": skimage.__version__,
    "timm": timm.__version__,
    "upstream_commit": UPSTREAM_COMMIT,
    "dataset_source": DATA_SOURCE,
}

environment_text = "\n".join(f"{key}={value}" for key, value in versions.items())
print(environment_text)
(RUN_ROOT / "environment.txt").write_text(environment_text, encoding="utf-8")
```

### Cell 7 - Save the exact command and configuration

```python
command = f"""python examples/musc_main.py \\
  --device 0 \\
  --data_path {DATA_ROOT} \\
  --dataset_name mvtec_ad \\
  --class_name bottle \\
  --backbone_name ViT-L-14-336 \\
  --pretrained openai \\
  --feature_layers 5 11 17 23 \\
  --img_resize 518 \\
  --divide_num 1 \\
  --r_list 1 3 5 \\
  --batch_size 4 \\
  --output_dir {OUTPUT_ROOT} \\
  --vis True \\
  --vis_type whole_norm \\
  --save_excel True"""

print(command)
(RUN_ROOT / "command.sh").write_text(
    "#!/usr/bin/env bash\nset -euo pipefail\n" + command + "\n",
    encoding="utf-8",
)
```

### Cell 7A - Resolve the Colab `datasets` package collision

Before running E00, apply this Colab compatibility fix. Colab may include the Hugging Face `datasets` package, which conflicts with MuSc's local `datasets` directory. The upstream entry point appends the repository root to `sys.path`, and the local directory does not contain `__init__.py`. Python can therefore resolve `datasets` to the external package and fail to import `datasets.mvtec`.

```python
from pathlib import Path
import subprocess
import sys

repo_root = Path("/content/MuSc")
entrypoint = repo_root / "examples" / "musc_main.py"
local_package = repo_root / "datasets" / "__init__.py"

source = entrypoint.read_bytes()
old_line = b"sys.path.append(os.getcwd())"
new_line = b"sys.path.insert(0, os.getcwd())"

if old_line in source:
    entrypoint.write_bytes(source.replace(old_line, new_line, 1))
elif new_line not in source:
    raise RuntimeError("The expected sys.path line was not found in musc_main.py")

package_marker = '\"\"\"Dataset loaders bundled with MuSc.\"\"\"\n'
if local_package.exists():
    assert local_package.read_text(encoding="utf-8") == package_marker, (
        f"Unexpected existing package marker: {local_package}"
    )
else:
    local_package.write_text(package_marker, encoding="utf-8")

diagnostic = subprocess.run(
    [
        sys.executable,
        "-c",
        "import datasets; print(datasets.__file__); "
        "import datasets.mvtec; print('Imported datasets.mvtec successfully')",
    ],
    cwd=repo_root,
    text=True,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    check=True,
)
print(diagnostic.stdout)

resolved_package = Path(diagnostic.stdout.splitlines()[0]).resolve()
assert repo_root.resolve() in resolved_package.parents, (
    f"Python still resolves an external datasets package: {resolved_package}"
)

git_status = subprocess.run(
    ["git", "status", "--short"],
    cwd=repo_root,
    text=True,
    stdout=subprocess.PIPE,
    check=True,
).stdout
git_diff = subprocess.run(
    ["git", "diff", "--", "examples/musc_main.py"],
    cwd=repo_root,
    text=True,
    stdout=subprocess.PIPE,
    check=True,
).stdout

(RUN_ROOT / "compatibility-fix.txt").write_text(
    "Reason: avoid collision with the Hugging Face datasets package.\n"
    "Effect on method: none; import resolution only.\n\n"
    + git_status
    + "\n"
    + git_diff,
    encoding="utf-8",
)
```

The diagnostic output must resolve `datasets` to `/content/MuSc/datasets/__init__.py`. This compatibility change only controls import resolution and does not modify the MuSc method.

### Cell 8 - Run E00

```python
import os
import subprocess
import time

started = time.time()
process = subprocess.run(
    ["bash", str(RUN_ROOT / "command.sh")],
    cwd="/content/MuSc",
    text=True,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
)
elapsed = time.time() - started

(RUN_ROOT / "console.log").write_text(process.stdout, encoding="utf-8")
(RUN_ROOT / "elapsed-seconds.txt").write_text(f"{elapsed:.3f}\n", encoding="utf-8")
print(process.stdout)
print(f"Elapsed time: {elapsed / 60:.2f} minutes")

if process.returncode != 0:
    raise RuntimeError(
        f"MuSc exited with code {process.returncode}. "
        f"Inspect {RUN_ROOT / 'console.log'} before changing the configuration."
    )
```

The first execution also downloads the official OpenAI CLIP weights. A Colab disconnection during this cell does not justify changing `divide_num`; first rerun from a clean runtime and inspect the saved log.

### Cell 9 - Verify and display results

```python
from pathlib import Path
from IPython.display import display
import pandas as pd

result_file = OUTPUT_ROOT / "mvtec_ad" / "ViT-L-14-336" / "imagesize518" / "results.xlsx"
assert result_file.is_file(), f"Missing result file: {result_file}"

results = pd.read_excel(result_file)
display(results)

image_files = sorted(
    path for path in (OUTPUT_ROOT / "mvtec_ad" / "ViT-L-14-336" / "imagesize518" / "bottle").rglob("*")
    if path.is_file()
)
print("Saved anomaly maps:", len(image_files))
print("Run directory:", RUN_ROOT)
```

### Cell 10 - Compare E00 with the official repository reference

```python
import pandas as pd

reference = pd.DataFrame([{
    "category": "bottle",
    "auroc_px": 98.48,
    "f1_px": 79.17,
    "ap_px": 83.04,
    "aupro": 96.10,
    "auroc_sp": 99.92,
    "f1_sp": 99.21,
    "ap_sp": 99.98,
}])

reproduced = pd.read_excel(result_file).iloc[[0]].copy()
reproduced = reproduced.rename(columns={reproduced.columns[0]: "category"})
reproduced.insert(0, "source", "reproduced_E00")
reference.insert(0, "source", "official_repository")
comparison = pd.concat([reference, reproduced], ignore_index=True)
display(comparison)
comparison.to_csv(RUN_ROOT / "reference-comparison.csv", index=False)
```

Small numerical differences are possible across GPU and library versions. If the difference is material, first check the commit, complete 83-image split, backbone, image size, feature layers, `r_list`, and `divide_num`. Preserve the original failed or discrepant run before trying another configuration.

## E00 Acceptance Checklist

- [ ] The GPU assertion passes.
- [ ] The dataset check finds 83 `bottle` test images.
- [ ] `dataset-source.txt` records the Hugging Face revision when the mirror is used.
- [ ] The compatibility diagnostic resolves `/content/MuSc/datasets/__init__.py`.
- [ ] `compatibility-fix.txt` records the Colab import compatibility change.
- [ ] The checked-out Git hash exactly matches the frozen commit.
- [ ] `console.log` contains no exception or CUDA out-of-memory error.
- [ ] `results.xlsx` contains one `bottle` row and one mean row.
- [ ] The seven required metrics are nonzero.
- [ ] Anomaly-map images are saved under the timestamped run directory.
- [ ] `environment.txt`, `upstream-commit.txt`, `command.sh`, and `console.log` exist in Drive.
- [ ] The reproduced results are compared with the official `bottle` reference rather than silently replaced by it.

## What to Return for Review

After E00 finishes, provide these four artifacts:

1. `results.xlsx`
2. `reference-comparison.csv`
3. `environment.txt`
4. `console.log`

The next decision will be based on those artifacts:

- If E00 is consistent with the reference, freeze the baseline and run all MVTec AD categories.
- If E00 completes but differs materially, diagnose the mismatch before scaling up.
- If E00 fails because of GPU memory, preserve the run and evaluate the smallest justified computational adjustment.

Do not implement top-k mean pooling until E00 is accepted. The extension must share the same baseline configuration so that the comparison remains controlled.

## Primary Sources

1. Xurui Li, Ziming Huang, Feng Xue, and Yu Zhou. [MuSc: Zero-Shot Industrial Anomaly Classification and Segmentation with Mutual Scoring of the Unlabeled Images](https://openreview.net/forum?id=AHgc5SMdtd). ICLR 2024.
2. [Official MuSc implementation](https://github.com/xrli-U/MuSc), frozen at commit [`b76b93d`](https://github.com/xrli-U/MuSc/commit/b76b93da8bd3096a99964a96ae29d46f197a0651).
3. [Official MVTec AD dataset page](https://www.mvtec.com/research-teaching/datasets/mvtec-ad).
