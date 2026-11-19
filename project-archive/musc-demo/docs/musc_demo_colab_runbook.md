# MuSc Classroom Demo - Google Colab Runbook

## Purpose and scope

This notebook launches the classroom demo in a fresh Google Colab runtime with
an NVIDIA T4 GPU. The demo has two independent paths:

- **Prepared Benchmark** reads immutable artifacts produced by completed MVTec
  AD or VisA experiments.
- **Live Inference** accepts one unlabeled set of 5-200 JPEG/PNG images from one
  industrial product category. The recommended range is 40-200 images.

The custom-upload path returns relative anomaly scores, heatmaps, overlays, and
a ranking within the uploaded set. It does not produce calibrated labels,
benchmark metrics, or medical conclusions.

Before class, run every cell in order and keep the final cell running. Store a
GitHub personal access token with read access to the private reproduction
repository in Colab Secrets under the name `GITHUB_TOKEN`. The notebook reads
the secret at runtime and never prints or writes it.

## Cell 1 - Confirm the GPU and define the frozen runtime

```python
from pathlib import Path
import os
import platform
import subprocess
import sys

import torch

PINNED_MUSC_COMMIT = "b76b93da8bd3096a99964a96ae29d46f197a0651"
REPRO_ROOT = Path("/content/cs2203-musc-reproduction")
UPSTREAM_ROOT = Path("/content/MuSc")
DEMO_CONFIG = REPRO_ROOT / "configs" / "demo.json"

assert torch.cuda.is_available(), (
    "CUDA GPU is unavailable. In Colab, select Runtime > Change runtime type > T4 GPU."
)

os.environ["MUSC_ROOT"] = str(UPSTREAM_ROOT)
os.environ["MUSC_EAGER_INIT"] = "0"
os.environ["MUSC_COMMIT"] = PINNED_MUSC_COMMIT

print("python:", sys.version)
print("platform:", platform.platform())
print("torch:", torch.__version__)
print("torchvision is intentionally kept from the Colab runtime")
print("cuda runtime:", torch.version.cuda)
print("gpu:", torch.cuda.get_device_name(0))
subprocess.run(["nvidia-smi"], check=True)
```

## Cell 2 - Clone the private reproduction repository

This cell uses a temporary `GIT_ASKPASS` script so the token is not placed in a
clone URL, notebook output, or committed file. Re-running the cell replaces only
the named ephemeral checkout under `/content`.

```python
from google.colab import userdata
import os
import shutil
import stat
import subprocess
import tempfile
from pathlib import Path

github_token = userdata.get("GITHUB_TOKEN")
assert github_token, "Add GITHUB_TOKEN to Colab Secrets and grant this notebook access."

if REPRO_ROOT.exists():
    shutil.rmtree(REPRO_ROOT)

with tempfile.TemporaryDirectory(prefix="musc-git-auth-") as auth_directory:
    askpass = Path(auth_directory) / "askpass.sh"
    askpass.write_text(
        "#!/usr/bin/env bash\n"
        "case \"$1\" in\n"
        "  *Username*) printf '%s\\n' \"$GIT_USERNAME\" ;;\n"
        "  *Password*) printf '%s\\n' \"$GIT_PASSWORD\" ;;\n"
        "esac\n",
        encoding="utf-8",
    )
    askpass.chmod(askpass.stat().st_mode | stat.S_IXUSR)
    clone_environment = os.environ.copy()
    clone_environment.update(
        {
            "GIT_ASKPASS": str(askpass),
            "GIT_TERMINAL_PROMPT": "0",
            "GIT_USERNAME": "x-access-token",
            "GIT_PASSWORD": github_token,
        }
    )
    subprocess.run(
        [
            "git",
            "clone",
            "https://github.com/DiiKhanh/cs2203-musc-reproduction.git",
            str(REPRO_ROOT),
        ],
        check=True,
        env=clone_environment,
    )

del github_token
assert (REPRO_ROOT / "musc_demo" / "app.py").is_file()
print("Reproduction checkout:", REPRO_ROOT)
```

## Cell 3 - Clone, pin, verify, and patch the official MuSc source

Commit verification happens before any model import or app launch. The patch
changes Python import precedence only; it does not alter CLIP, LNAMD, MSM, or
RsCIN calculations.

```bash
set -euo pipefail
rm -rf /content/MuSc
git clone https://github.com/xrli-U/MuSc.git /content/MuSc
git -C /content/MuSc checkout --detach "$MUSC_COMMIT"
ACTUAL_COMMIT="$(git -C /content/MuSc rev-parse HEAD)"
test "$ACTUAL_COMMIT" = "b76b93da8bd3096a99964a96ae29d46f197a0651"
git -C /content/MuSc apply /content/cs2203-musc-reproduction/patches/colab-import-compatibility.patch
printf 'Verified MuSc commit: %s\n' "$ACTUAL_COMMIT"
```

## Cell 4 - Install demo dependencies without replacing PyTorch

`requirements-demo.txt` explicitly keeps PyTorch and torchvision out of the
installation set. The assertions below guard against an accidental runtime
replacement.

```python
import subprocess
import sys
import torch

torch_before = torch.__version__
subprocess.run(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "-r",
        str(REPRO_ROOT / "requirements-demo.txt"),
    ],
    check=True,
)

import torch as torch_after_install

assert torch_after_install.__version__ == torch_before
print("Retained Colab torch:", torch_after_install.__version__)
```

## Cell 5 - Record the resolved environment

```python
import gradio
import numpy
import pandas
import PIL
import torch
import torchvision

versions = {
    "torch": torch.__version__,
    "torchvision": torchvision.__version__,
    "cuda_runtime": torch.version.cuda,
    "gpu": torch.cuda.get_device_name(0),
    "gradio": gradio.__version__,
    "numpy": numpy.__version__,
    "pandas": pandas.__version__,
    "pillow": PIL.__version__,
    "musc_commit": PINNED_MUSC_COMMIT,
}
print("\n".join(f"{name}={value}" for name, value in versions.items()))
```

## Cell 6 - Optionally mount prepared benchmark artifacts

Set `USE_PREPARED_RESULTS` to `True` only when the external bundle is available.
The relative path is deliberately generic and contains no account-specific
Google Drive location. The bundle must contain `manifest.json` at its root.

```python
from pathlib import Path
import os

USE_PREPARED_RESULTS = False
PREPARED_RELATIVE_PATH = Path("CS2203_MuSc/prepared")

if USE_PREPARED_RESULTS:
    from google.colab import drive

    drive.mount("/content/drive")
    prepared_root = Path("/content/drive/MyDrive") / PREPARED_RELATIVE_PATH
    assert (prepared_root / "manifest.json").is_file(), (
        f"Prepared manifest is missing: {prepared_root / 'manifest.json'}"
    )
    os.environ["MUSC_PREPARED_ROOT"] = str(prepared_root)
    print("Prepared Benchmark enabled:", prepared_root)
else:
    os.environ.pop("MUSC_PREPARED_ROOT", None)
    print("Prepared Benchmark not mounted. Live Inference can still run.")
```

## Cell 7 - Preflight the pinned engine and fallback state

The app starts without loading the backbone so the share URL and Prepared
Benchmark remain available even if model initialization later fails. Initialize
MuSc from the app only when Live Inference is needed.

```python
import json
import os
import subprocess

resolved_commit = subprocess.check_output(
    ["git", "-C", str(UPSTREAM_ROOT), "rev-parse", "HEAD"],
    text=True,
).strip()
assert resolved_commit == PINNED_MUSC_COMMIT

configuration = json.loads(DEMO_CONFIG.read_text(encoding="utf-8"))
assert configuration["upstream_commit"] == PINNED_MUSC_COMMIT
assert configuration["rscin_neighbors"] == [1, 2, 3]

if "MUSC_PREPARED_ROOT" not in os.environ:
    print("Fallback note: Prepared Benchmark is unavailable until its bundle is mounted.")
print("Fallback note: if live inference fails, keep the prepared tab open and reduce the batch.")
print("Preflight complete. The next cell opens the app; initialize MuSc inside it.")
```

## Cell 8 - Launch the app before initializing MuSc

The cell blocks while the app is running. Stop it to shut down the temporary
public link. Prepared Benchmark is available immediately; initialize MuSc from
the app only when Live Inference is needed.

```python
import os
import sys

os.environ["MUSC_ROOT"] = str(UPSTREAM_ROOT)
os.environ["MUSC_EAGER_INIT"] = "0"

if str(REPRO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPRO_ROOT))

print("[1/4] Importing the MuSc demo...", flush=True)
from musc_demo.app import (
    build_app,
    default_services,
    demo_theme,
    theme_css_path,
)

print("[2/4] Loading configuration and Prepared Benchmark...", flush=True)
services = default_services()

print("[3/4] Building the interface...", flush=True)
demo = build_app(services)

print("[4/4] Starting Gradio and creating a share URL...", flush=True)
demo.launch(
    share=True,
    debug=True,
    show_error=True,
    css_paths=theme_css_path(),
    theme=demo_theme(),
)
```

## Presentation fallback

- If CUDA is unavailable, change the Colab runtime to T4 and restart from Cell
  1. Prepared Benchmark remains the presentation fallback when mounted.
- If model initialization or live inference fails, do not expose the notebook
  traceback in class. Explain the failed pipeline stage in the app, reduce the
  batch when memory is insufficient, and continue with Prepared Benchmark.
- If prepared artifacts are unavailable, the app states that explicitly. Do
  not replace missing reproduced metrics with values copied from the paper.
- Only accept an instructor-provided batch when it satisfies the displayed
  one-category industrial input contract.
