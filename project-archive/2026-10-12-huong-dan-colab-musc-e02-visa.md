# Hướng dẫn chạy MuSc E02 trên toàn bộ VisA bằng Google Colab

## 1. Mục tiêu

E02 kiểm tra khả năng tái lập MuSc trên benchmark thứ hai là VisA. Notebook sẽ:

1. Tải VisA từ nguồn Amazon S3 chính thức.
2. Tạo `meta.json` bằng đúng script `datasets/visa_preprocess.py` của MuSc.
3. Chạy riêng 12 category để có thể resume sau khi Colab ngắt kết nối.
4. Giữ nguyên baseline của E01: OpenAI CLIP `ViT-L-14-336`, image size 518, feature layers `[5, 11, 17, 23]`, `r_list = [1, 3, 5]`, batch size 4 và `divide_num = 1`.
5. Tổng hợp kết quả và so sánh với bảng VisA trong repository MuSc.
6. Đóng gói artifact thành ZIP để kiểm tra.

Nếu runtime E01Q hiện tại vẫn còn GPU T4, có thể thêm một section `E02 - VisA` vào cuối notebook đó và chạy tiếp để tận dụng GPU đã được Colab cấp. E02-1 tạo Run ID và thư mục riêng nên không ghi đè E01Q. E02-4 sẽ clone lại `/content/MuSc`, nhưng toàn bộ artifact E01Q đã nằm an toàn trên Drive.

Nếu runtime E01Q đã mất, tạo một notebook Colab mới và đặt tên gợi ý:

```text
CS2203_MuSc_E02_VisA.ipynb
```

Mỗi khối Python bên dưới là một cell riêng. Không chạy lại các cell E01 hoặc E01Q sau khi bắt đầu E02 và không đổi cấu hình trong lúc E02 đang chạy.

Thời gian dự kiến phụ thuộc quota và runtime Colab. Nên dành một phiên GPU dài, nhưng notebook có thể resume theo từng category.

## 2. Cell E02-1 - Mount Drive và tạo hoặc resume Run ID

```python
from google.colab import drive
drive.mount("/content/drive")

from datetime import datetime, timezone
from pathlib import Path
import json
import os

UPSTREAM_COMMIT = "b76b93da8bd3096a99964a96ae29d46f197a0651"
PROJECT_ROOT = Path("/content/drive/MyDrive/CS2203_MuSc")
RUNS_ROOT = PROJECT_ROOT / "runs"
ACTIVE_RUN_FILE = PROJECT_ROOT / "active-e02-run.txt"

CATEGORIES = [
    "candle", "capsules", "cashew", "chewinggum", "fryum",
    "macaroni1", "macaroni2", "pcb1", "pcb2", "pcb3",
    "pcb4", "pipe_fryum",
]

PROJECT_ROOT.mkdir(parents=True, exist_ok=True)
RUNS_ROOT.mkdir(parents=True, exist_ok=True)

if ACTIVE_RUN_FILE.is_file():
    candidate_run_id = ACTIVE_RUN_FILE.read_text(encoding="utf-8").strip()
    candidate_root = RUNS_ROOT / candidate_run_id
    if candidate_root.is_dir() and not (candidate_root / "E02_AGGREGATED.json").exists():
        RUN_ID = candidate_run_id
        print("Tiếp tục E02 Run ID đang dở:", RUN_ID)
    else:
        ACTIVE_RUN_FILE.unlink()
        RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S_E02_visa_all_clip_l14_336_baseline")
        ACTIVE_RUN_FILE.write_text(RUN_ID + "\n", encoding="utf-8")
        print("Tạo E02 Run ID mới:", RUN_ID)
else:
    RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S_E02_visa_all_clip_l14_336_baseline")
    ACTIVE_RUN_FILE.write_text(RUN_ID + "\n", encoding="utf-8")
    print("Tạo E02 Run ID mới:", RUN_ID)

RUN_ROOT = RUNS_ROOT / RUN_ID
CATEGORY_ROOT = RUN_ROOT / "categories"
SESSION_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
SESSION_ROOT = RUN_ROOT / "sessions" / SESSION_ID

CATEGORY_ROOT.mkdir(parents=True, exist_ok=True)
SESSION_ROOT.mkdir(parents=True, exist_ok=False)

os.environ["MUSC_COMMIT"] = UPSTREAM_COMMIT
os.environ["MUSC_E02_RUN_ROOT"] = str(RUN_ROOT)

(RUN_ROOT / "run-id.txt").write_text(RUN_ID + "\n", encoding="utf-8")

print("Run root:", RUN_ROOT)
print("Session root:", SESSION_ROOT)
```

Khi Colab bị ngắt, `active-e02-run.txt` giúp cell này tìm lại Run ID. Không xóa file đó khi E02 chưa tổng hợp xong.

## 3. Cell E02-2 - Kiểm tra GPU và cài dependency

```python
import platform
import subprocess
import sys

subprocess.run(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "--upgrade",
        "huggingface_hub",
        "ftfy==6.1.3",
        "openpyxl==3.1.2",
        "timm==0.9.12",
    ],
    check=True,
)

import cv2
import ftfy
import openpyxl
import sklearn
import skimage
import timm
import torch
import torchvision

assert torch.cuda.is_available(), "GPU chưa được bật hoặc Colab chưa cấp GPU."

versions = {
    "python": sys.version,
    "platform": platform.platform(),
    "torch": torch.__version__,
    "torchvision": torchvision.__version__,
    "cuda_runtime": torch.version.cuda,
    "gpu": torch.cuda.get_device_name(0),
    "opencv": cv2.__version__,
    "ftfy": ftfy.__version__,
    "openpyxl": openpyxl.__version__,
    "scikit_learn": sklearn.__version__,
    "scikit_image": skimage.__version__,
    "timm": timm.__version__,
    "upstream_commit": UPSTREAM_COMMIT,
}

environment_text = "\n".join(f"{key}={value}" for key, value in versions.items())
(SESSION_ROOT / "environment.txt").write_text(environment_text + "\n", encoding="utf-8")
print(environment_text)
subprocess.run(["nvidia-smi"], check=True)
```

Nếu cell này báo không có GPU, dừng lại. Không chạy E02 bằng CPU.

## 4. Cell E02-3 - Tải, kiểm tra và giải nén VisA

Dataset chính thức có dung lượng khoảng 2 GB. Cell này lưu một bản TAR thông thường trên Drive để những runtime sau không phải tải lại. Dataset đã giải nén vẫn nằm trong `/content` và phải được dựng lại sau mỗi runtime mới.

```python
from datetime import datetime
import hashlib
import shutil
import subprocess

VISA_URL = "https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar"
DRIVE_DATASET_CACHE = PROJECT_ROOT / "cache" / "visa"
DRIVE_DATASET_CACHE.mkdir(parents=True, exist_ok=True)
DRIVE_ARCHIVE = DRIVE_DATASET_CACHE / "VisA_20220922.tar"
LOCAL_ARCHIVE = Path("/content/VisA_20220922.tar")
LOCAL_PARTIAL = Path("/content/VisA_20220922.tar.partial")

def valid_tar(path):
    candidate = Path(path)
    if not candidate.is_file() or candidate.stat().st_size < 1_000_000_000:
        return False, "File thiếu hoặc nhỏ bất thường."
    result = subprocess.run(
        ["tar", "-tf", str(candidate)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    return result.returncode == 0, result.stderr.strip()

if DRIVE_ARCHIVE.is_file():
    print("Đang sao chép VisA archive từ Drive xuống /content...")
    shutil.copy2(DRIVE_ARCHIVE, LOCAL_ARCHIVE)
    cached_valid, cached_message = valid_tar(LOCAL_ARCHIVE)
    print("Drive cache valid:", cached_valid)
    if not cached_valid:
        invalid_name = DRIVE_ARCHIVE.with_name(
            DRIVE_ARCHIVE.name + ".invalid-" + datetime.now().strftime("%Y%m%d_%H%M%S")
        )
        DRIVE_ARCHIVE.replace(invalid_name)
        LOCAL_ARCHIVE.unlink(missing_ok=True)
        print("Đã giữ cache lỗi tại:", invalid_name)
        print("Lý do:", cached_message)

if not LOCAL_ARCHIVE.is_file():
    LOCAL_PARTIAL.unlink(missing_ok=True)
    print("Đang tải VisA từ Amazon S3 chính thức...")
    subprocess.run(
        [
            "curl", "-L", "--fail", "--retry", "5",
            "--output", str(LOCAL_PARTIAL), VISA_URL,
        ],
        check=True,
    )
    downloaded_valid, downloaded_message = valid_tar(LOCAL_PARTIAL)
    print("Downloaded archive valid:", downloaded_valid)
    print("Archive check:", downloaded_message or "OK")
    assert downloaded_valid, "VisA archive tải xuống không hợp lệ."
    LOCAL_PARTIAL.replace(LOCAL_ARCHIVE)

local_valid, local_message = valid_tar(LOCAL_ARCHIVE)
assert local_valid, f"Local TAR không hợp lệ: {local_message}"

if not DRIVE_ARCHIVE.is_file() or DRIVE_ARCHIVE.stat().st_size != LOCAL_ARCHIVE.stat().st_size:
    print("Đang lưu archive hợp lệ vào Drive cache...")
    shutil.copy2(LOCAL_ARCHIVE, DRIVE_ARCHIVE)

sha256 = hashlib.sha256()
with LOCAL_ARCHIVE.open("rb") as stream:
    for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
        sha256.update(chunk)
archive_sha256 = sha256.hexdigest()

extract_root = Path("/content/visa-extract")
if extract_root.exists():
    shutil.rmtree(extract_root)
extract_root.mkdir(parents=True, exist_ok=False)

print("Đang giải nén VisA...")
subprocess.run(
    ["tar", "-xf", str(LOCAL_ARCHIVE), "-C", str(extract_root)],
    check=True,
)

split_files = list(extract_root.rglob("split_csv/1cls.csv"))
assert len(split_files) == 1, f"Không xác định được duy nhất VisA root: {split_files}"
DATA_ROOT = split_files[0].parent.parent

for category in CATEGORIES:
    required_paths = [
        DATA_ROOT / category / "Data" / "Images" / "Normal",
        DATA_ROOT / category / "Data" / "Images" / "Anomaly",
        DATA_ROOT / category / "Data" / "Masks" / "Anomaly",
        DATA_ROOT / category / "image_anno.csv",
    ]
    for required in required_paths:
        assert required.exists(), f"Thiếu dữ liệu: {required}"

DATA_SOURCE = VISA_URL
dataset_source_text = (
    f"source={DATA_SOURCE}\n"
    "archive=VisA_20220922.tar\n"
    f"sha256={archive_sha256}\n"
    f"resolved_root={DATA_ROOT}\n"
)
dataset_source_path = RUN_ROOT / "dataset-source.txt"
if dataset_source_path.exists():
    previous = dataset_source_path.read_text(encoding="utf-8").splitlines()
    assert previous[0] == f"source={DATA_SOURCE}", "Nguồn dataset thay đổi khi resume."
    assert previous[2] == f"sha256={archive_sha256}", "SHA-256 dataset thay đổi khi resume."
else:
    dataset_source_path.write_text(dataset_source_text, encoding="utf-8")

print("Archive SHA-256:", archive_sha256)
print("Dataset root:", DATA_ROOT)
print("Đã xác nhận đủ 12 category VisA.")
```

Chỉ tiếp tục khi cell xác nhận đủ 12 category.

## 5. Cell E02-4 - Clone MuSc, sửa import và tạo `meta.json`

```python
import shutil
import subprocess

REPO_ROOT = Path("/content/MuSc")
if REPO_ROOT.exists():
    shutil.rmtree(REPO_ROOT)

subprocess.run(
    ["git", "clone", "https://github.com/xrli-U/MuSc.git", str(REPO_ROOT)],
    check=True,
)
subprocess.run(
    ["git", "checkout", "--detach", UPSTREAM_COMMIT],
    cwd=REPO_ROOT,
    check=True,
)

actual_commit = subprocess.run(
    ["git", "rev-parse", "HEAD"],
    cwd=REPO_ROOT,
    text=True,
    stdout=subprocess.PIPE,
    check=True,
).stdout.strip()
assert actual_commit == UPSTREAM_COMMIT

entrypoint = REPO_ROOT / "examples" / "musc_main.py"
local_package = REPO_ROOT / "datasets" / "__init__.py"
source = entrypoint.read_bytes()
old_line = b"sys.path.append(os.getcwd())"
new_line = b"sys.path.insert(0, os.getcwd())"

if old_line in source:
    entrypoint.write_bytes(source.replace(old_line, new_line, 1))
elif new_line not in source:
    raise RuntimeError("Không tìm thấy dòng sys.path mong đợi trong musc_main.py")

local_package.touch(exist_ok=True)

diagnostic = subprocess.run(
    [
        sys.executable,
        "-c",
        "import datasets; print(datasets.__file__); "
        "import datasets.visa; print('Đã import datasets.visa thành công')",
    ],
    cwd=REPO_ROOT,
    text=True,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    check=True,
)
print(diagnostic.stdout)

resolved_package = Path(diagnostic.stdout.splitlines()[0]).resolve()
assert REPO_ROOT.resolve() in resolved_package.parents

repo_data_root = REPO_ROOT / "data"
repo_data_root.mkdir(parents=True, exist_ok=True)
visa_link = repo_data_root / "visa"
if visa_link.is_symlink() or visa_link.is_file():
    visa_link.unlink()
elif visa_link.is_dir():
    shutil.rmtree(visa_link)
visa_link.symlink_to(DATA_ROOT, target_is_directory=True)

subprocess.run(
    [sys.executable, "datasets/visa_preprocess.py"],
    cwd=REPO_ROOT,
    check=True,
)

meta_path = DATA_ROOT / "meta.json"
assert meta_path.is_file(), "visa_preprocess.py chưa tạo meta.json"
meta = json.loads(meta_path.read_text(encoding="utf-8"))
assert set(meta) == {"train", "test"}

test_counts = {}
for category in CATEGORIES:
    assert category in meta["test"], f"meta.json thiếu category {category}"
    test_counts[category] = len(meta["test"][category])
    assert test_counts[category] > 0, f"Test split rỗng: {category}"
    for sample in meta["test"][category]:
        assert (DATA_ROOT / sample["img_path"]).is_file(), sample["img_path"]
        if sample["mask_path"]:
            assert (DATA_ROOT / sample["mask_path"]).is_file(), sample["mask_path"]

git_status = subprocess.run(
    ["git", "status", "--short"],
    cwd=REPO_ROOT,
    text=True,
    stdout=subprocess.PIPE,
    check=True,
).stdout
git_diff = subprocess.run(
    ["git", "diff", "--", "examples/musc_main.py"],
    cwd=REPO_ROOT,
    text=True,
    stdout=subprocess.PIPE,
    check=True,
).stdout

(SESSION_ROOT / "compatibility-fix.txt").write_text(
    "Reason: avoid collision with the Hugging Face datasets package.\n"
    "Effect on method: none; import resolution only.\n\n"
    + git_status + "\n" + git_diff,
    encoding="utf-8",
)
(RUN_ROOT / "upstream-commit.txt").write_text(actual_commit + "\n", encoding="utf-8")
(RUN_ROOT / "visa-test-counts.json").write_text(
    json.dumps(test_counts, indent=2) + "\n",
    encoding="utf-8",
)

print("VisA test counts:", test_counts)
print("Tổng số test image:", sum(test_counts.values()))
print("Đã tạo và kiểm tra meta.json.")
```

## 6. Cell E02-4M - Áp dụng bản vá giảm CPU RAM

Đây là cùng bản vá đã dùng và kiểm tra trong E01Q. Bản vá giải phóng tensor hết vòng đời và bỏ hai bản sao NumPy không cần thiết. Nó không thay đổi công thức MuSc hoặc metric.

```python
import subprocess

musc_path = REPO_ROOT / "models" / "musc.py"
source = musc_path.read_bytes()
line_ending = b"\r\n" if b"\r\n" in source else b"\n"

def encode_block(text):
    return text.encode("utf-8").replace(b"\n", line_ending)

def replace_once(data, old_text, new_text, label):
    old = encode_block(old_text)
    new = encode_block(new_text)
    if new in data:
        print(f"SKIP {label}: đã áp dụng")
        return data
    count = data.count(old)
    assert count == 1, f"Không tìm thấy duy nhất marker {label}: count={count}"
    print(f"APPLY {label}")
    return data.replace(old, new, 1)

source = replace_once(source, "import cv2\n", "import cv2\nimport gc\n", "import gc")
source = replace_once(
    source,
    "        anomaly_maps = anomaly_maps.cpu().numpy()\n"
    "        torch.cuda.empty_cache()\n",
    "        anomaly_maps = anomaly_maps.cpu().numpy()\n"
    "        # Release feature tensors before allocating flattened metric arrays.\n"
    "        del patch_tokens_list, Z_layers, Z, features, patch_tokens\n"
    "        del anomaly_maps_l, anomaly_maps_msm, anomaly_maps_iter\n"
    "        del image_info, image, input_image, test_dataloader, test_dataset, LNAMD_r\n"
    "        gc.collect()\n"
    "        torch.cuda.empty_cache()\n",
    "release expired feature tensors",
)
source = replace_once(
    source,
    "        ac_score = np.array(anomaly_maps).reshape(B, -1).max(-1)\n",
    "        ac_score = anomaly_maps.reshape(B, -1).max(-1)\n",
    "remove anomaly-map copy for image score",
)
source = replace_once(
    source,
    "        pr_px = np.array(anomaly_maps)\n",
    "        pr_px = anomaly_maps\n",
    "remove anomaly-map copy for pixel metrics",
)
musc_path.write_bytes(source)

memory_diff = subprocess.run(
    ["git", "diff", "--", "models/musc.py"],
    cwd=REPO_ROOT,
    text=True,
    stdout=subprocess.PIPE,
    check=True,
).stdout
assert "gc.collect()" in memory_diff
assert "ac_score = anomaly_maps.reshape" in memory_diff
assert "pr_px = anomaly_maps" in memory_diff

(SESSION_ROOT / "cpu-memory-patch.diff").write_text(memory_diff, encoding="utf-8")
print(memory_diff)
```

## 7. Cell E02-5 - Khôi phục CLIP weights

```python
import shutil

LOCAL_CLIP_CACHE = Path.home() / ".cache" / "clip"
DRIVE_CLIP_CACHE = PROJECT_ROOT / "cache" / "clip"
LOCAL_CLIP_CACHE.mkdir(parents=True, exist_ok=True)
DRIVE_CLIP_CACHE.mkdir(parents=True, exist_ok=True)

restored_files = []
for source_file in DRIVE_CLIP_CACHE.glob("*"):
    if source_file.is_file():
        target_file = LOCAL_CLIP_CACHE / source_file.name
        if not target_file.exists() or target_file.stat().st_size != source_file.stat().st_size:
            shutil.copy2(source_file, target_file)
        restored_files.append(target_file.name)

if restored_files:
    print("Đã khôi phục CLIP cache:", restored_files)
else:
    print("Chưa có CLIP cache trong Drive. Category đầu tiên sẽ tải weights khoảng 934 MB.")
```

## 8. Cell E02-6 - Lưu cấu hình baseline

```python
BASELINE_CONFIG = {
    "dataset_name": "visa",
    "backbone_name": "ViT-L-14-336",
    "pretrained": "openai",
    "feature_layers": [5, 11, 17, 23],
    "img_resize": 518,
    "divide_num": 1,
    "r_list": [1, 3, 5],
    "batch_size": 4,
    "vis": False,
    "vis_type": "whole_norm",
    "save_excel": True,
    "seed_in_upstream_entrypoint": 42,
}

config_path = RUN_ROOT / "baseline-config.json"
config_text = json.dumps(BASELINE_CONFIG, indent=2, sort_keys=True) + "\n"
if config_path.exists():
    assert config_path.read_text(encoding="utf-8") == config_text, "Baseline config đã thay đổi."
else:
    config_path.write_text(config_text, encoding="utf-8")

print(config_text)
```

## 9. Cell E02-7 - Định nghĩa hàm chạy một category

```python
from datetime import datetime, timezone
import json
import math
import pandas as pd
import shlex
import shutil
import subprocess
import time

METRIC_COLUMNS = [
    "auroc_px", "f1_px", "ap_px", "aupro",
    "auroc_sp", "f1_sp", "ap_sp",
]

def sync_clip_cache_to_drive():
    DRIVE_CLIP_CACHE.mkdir(parents=True, exist_ok=True)
    for source_file in LOCAL_CLIP_CACHE.glob("*"):
        if source_file.is_file():
            target_file = DRIVE_CLIP_CACHE / source_file.name
            if not target_file.exists() or target_file.stat().st_size != source_file.stat().st_size:
                shutil.copy2(source_file, target_file)

def successful_attempt(category):
    category_dir = CATEGORY_ROOT / category
    success_files = sorted(category_dir.glob("attempt_*/SUCCESS.json"))
    return success_files[-1].parent if success_files else None

def run_category(category):
    assert category in CATEGORIES, f"Category không hợp lệ: {category}"

    previous_success = successful_attempt(category)
    if previous_success is not None:
        metrics = json.loads((previous_success / "metrics.json").read_text(encoding="utf-8"))
        print(f"SKIP {category}: đã thành công tại {previous_success}")
        return metrics

    category_dir = CATEGORY_ROOT / category
    category_dir.mkdir(parents=True, exist_ok=True)
    attempt_number = len(list(category_dir.glob("attempt_*"))) + 1
    attempt_root = category_dir / f"attempt_{attempt_number:02d}"
    output_root = attempt_root / "output"
    attempt_root.mkdir(parents=True, exist_ok=False)

    command = [
        sys.executable,
        "examples/musc_main.py",
        "--device", "0",
        "--data_path", str(DATA_ROOT),
        "--dataset_name", "visa",
        "--class_name", category,
        "--backbone_name", "ViT-L-14-336",
        "--pretrained", "openai",
        "--feature_layers", "5", "11", "17", "23",
        "--img_resize", "518",
        "--divide_num", "1",
        "--r_list", "1", "3", "5",
        "--batch_size", "4",
        "--output_dir", str(output_root),
        "--vis", "False",
        "--vis_type", "whole_norm",
        "--save_excel", "True",
    ]

    (attempt_root / "command.sh").write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n" + shlex.join(command) + "\n",
        encoding="utf-8",
    )

    started_at = datetime.now(timezone.utc)
    started = time.time()
    log_path = attempt_root / "console.log"

    print(f"START {category}, attempt {attempt_number:02d}")
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
        )
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
        return_code = process.wait()

    elapsed_seconds = time.time() - started
    sync_clip_cache_to_drive()

    status = {
        "category": category,
        "attempt": attempt_number,
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": elapsed_seconds,
        "return_code": return_code,
        "gpu": torch.cuda.get_device_name(0),
    }

    if return_code != 0:
        (attempt_root / "FAILED.json").write_text(
            json.dumps(status, indent=2) + "\n",
            encoding="utf-8",
        )
        raise RuntimeError(f"{category} thất bại. Xem log: {log_path}")

    result_file = (
        output_root / "visa" / "ViT-L-14-336" /
        "imagesize518" / "results.xlsx"
    )
    assert result_file.is_file(), f"Thiếu file kết quả: {result_file}"

    frame = pd.read_excel(result_file)
    frame = frame.rename(columns={frame.columns[0]: "category"})
    row = frame.iloc[0]
    assert row["category"] == category

    metrics = {"category": category}
    metrics.update({column: float(row[column]) for column in METRIC_COLUMNS})
    for column in METRIC_COLUMNS:
        assert math.isfinite(metrics[column]) and 0 <= metrics[column] <= 100
    metrics["elapsed_seconds"] = elapsed_seconds
    metrics["attempt_root"] = str(attempt_root)

    (attempt_root / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n",
        encoding="utf-8",
    )
    (attempt_root / "SUCCESS.json").write_text(
        json.dumps(status, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"DONE {category}: {elapsed_seconds / 60:.2f} phút")
    print({column: round(metrics[column], 4) for column in METRIC_COLUMNS})
    return metrics
```

## 10. Cell E02-8 - Smoke test bằng `candle`

```python
CANDLE_REFERENCE = {
    "auroc_sp": 96.55,
    "f1_sp": 91.26,
    "ap_sp": 96.45,
    "auroc_px": 99.36,
    "f1_px": 39.56,
    "ap_px": 28.36,
    "aupro": 97.62,
}

candle_metrics = run_category("candle")
candle_deltas = {
    metric: candle_metrics[metric] - CANDLE_REFERENCE[metric]
    for metric in METRIC_COLUMNS
}

print("Chênh lệch candle với repository:")
for metric, delta in candle_deltas.items():
    print(f"  {metric}: {delta:+.6f}")

(RUN_ROOT / "candle-smoke-check.json").write_text(
    json.dumps(
        {
            "category": "candle",
            "deltas_from_repository": candle_deltas,
            "status": "completed",
        },
        indent=2,
    ) + "\n",
    encoding="utf-8",
)
```

Nếu `candle` tạo `SUCCESS.json` và đủ bảy metric hữu hạn trong khoảng 0 đến 100, có thể tiếp tục. Sai lệch sẽ được phân tích sau khi đủ 12 category, không sửa kết quả để ép khớp bảng repository.

## 11. Cell E02-8R - Kiểm tra tiến độ sau khi runtime bị ngắt

```python
from collections import deque

progress_rows = []
for category in CATEGORIES:
    category_dir = CATEGORY_ROOT / category
    attempts = sorted(category_dir.glob("attempt_*"))
    success = successful_attempt(category)

    if success is not None:
        status = "SUCCESS"
        latest = success
    elif attempts:
        latest = attempts[-1]
        status = "FAILED" if (latest / "FAILED.json").is_file() else "INTERRUPTED"
    else:
        latest = None
        status = "NOT_STARTED"

    progress_rows.append(
        {
            "category": category,
            "status": status,
            "attempts": len(attempts),
            "latest_attempt": str(latest) if latest else "",
        }
    )

progress = pd.DataFrame(progress_rows)
display(progress)

interrupted = progress.loc[progress["status"] == "INTERRUPTED", "category"].tolist()
remaining = progress.loc[progress["status"] != "SUCCESS", "category"].tolist()
print("Category bị gián đoạn:", interrupted)
print("Category còn lại:", remaining)

for category in interrupted:
    latest_attempt = Path(
        progress.loc[progress["category"] == category, "latest_attempt"].iloc[0]
    )
    log_path = latest_attempt / "console.log"
    if log_path.is_file():
        with log_path.open("r", encoding="utf-8", errors="replace") as stream:
            tail = list(deque(stream, maxlen=12))
        print(f"\n===== Cuối log của {category} =====")
        print("".join(tail))
```

## 12. Cell E02-9 - Chạy hoặc resume toàn bộ 12 category

```python
all_metrics = {}

for index, category in enumerate(CATEGORIES, start=1):
    print(f"\n===== [{index:02d}/{len(CATEGORIES)}] {category} =====")
    all_metrics[category] = run_category(category)

print("Đã hoàn thành đủ 12 category VisA.")
```

Nếu Colab ngắt kết nối:

1. Mở runtime GPU mới.
2. Chạy lại E02-1 đến E02-7, gồm E02-4M.
3. Chạy E02-8R để xem tiến độ.
4. Chạy E02-9. Category có `SUCCESS.json` sẽ được bỏ qua.

Giữ `divide_num = 1`. Việc tăng `divide_num` làm thay đổi tập ảnh tham gia mutual scoring và tạo một cấu hình khác.

## 13. Cell E02-10 - Tổng hợp kết quả

```python
records = []
missing_categories = []

for category in CATEGORIES:
    attempt_root = successful_attempt(category)
    if attempt_root is None:
        missing_categories.append(category)
        continue
    records.append(json.loads((attempt_root / "metrics.json").read_text(encoding="utf-8")))

assert not missing_categories, f"Chưa hoàn thành: {missing_categories}"
assert len(records) == 12

reproduced = pd.DataFrame(records)
reproduced["category"] = pd.Categorical(
    reproduced["category"], categories=CATEGORIES, ordered=True
)
reproduced = reproduced.sort_values("category").reset_index(drop=True)

mean_row = {"category": "mean"}
for column in METRIC_COLUMNS:
    mean_row[column] = reproduced[column].mean()
mean_row["elapsed_seconds"] = reproduced["elapsed_seconds"].sum()
mean_row["attempt_root"] = ""

reproduced_with_mean = pd.concat(
    [reproduced, pd.DataFrame([mean_row])], ignore_index=True
)

reproduced_with_mean.to_csv(RUN_ROOT / "visa_reproduced.csv", index=False)
reproduced_with_mean.to_excel(RUN_ROOT / "visa_reproduced.xlsx", index=False)

display(reproduced_with_mean)
print("Tổng thời gian category subprocess:", reproduced["elapsed_seconds"].sum() / 60, "phút")
```

## 14. Cell E02-11 - So sánh với bảng VisA của repository

```python
reference_records = [
    {"category": "candle", "auroc_sp": 96.55, "f1_sp": 91.26, "ap_sp": 96.45, "auroc_px": 99.36, "f1_px": 39.56, "ap_px": 28.36, "aupro": 97.62},
    {"category": "capsules", "auroc_sp": 88.62, "f1_sp": 86.43, "ap_sp": 93.77, "auroc_px": 98.71, "f1_px": 50.85, "ap_px": 43.90, "aupro": 88.20},
    {"category": "cashew", "auroc_sp": 98.54, "f1_sp": 95.57, "ap_sp": 99.30, "auroc_px": 99.33, "f1_px": 74.88, "ap_px": 77.63, "aupro": 94.30},
    {"category": "chewinggum", "auroc_sp": 98.42, "f1_sp": 96.45, "ap_sp": 99.30, "auroc_px": 99.54, "f1_px": 61.33, "ap_px": 61.21, "aupro": 88.39},
    {"category": "fryum", "auroc_sp": 98.64, "f1_sp": 97.44, "ap_sp": 99.43, "auroc_px": 99.43, "f1_px": 58.13, "ap_px": 50.43, "aupro": 94.38},
    {"category": "macaroni1", "auroc_sp": 89.33, "f1_sp": 82.76, "ap_sp": 88.64, "auroc_px": 99.51, "f1_px": 21.90, "ap_px": 15.25, "aupro": 96.37},
    {"category": "macaroni2", "auroc_sp": 68.03, "f1_sp": 69.96, "ap_sp": 67.37, "auroc_px": 97.14, "f1_px": 11.06, "ap_px": 3.91, "aupro": 88.84},
    {"category": "pcb1", "auroc_sp": 89.28, "f1_sp": 84.36, "ap_sp": 89.89, "auroc_px": 99.50, "f1_px": 80.49, "ap_px": 88.36, "aupro": 92.76},
    {"category": "pcb2", "auroc_sp": 93.20, "f1_sp": 88.66, "ap_sp": 94.46, "auroc_px": 97.39, "f1_px": 34.38, "ap_px": 21.86, "aupro": 86.06},
    {"category": "pcb3", "auroc_sp": 93.52, "f1_sp": 86.92, "ap_sp": 93.48, "auroc_px": 98.05, "f1_px": 40.23, "ap_px": 41.03, "aupro": 92.32},
    {"category": "pcb4", "auroc_sp": 98.43, "f1_sp": 92.89, "ap_sp": 98.47, "auroc_px": 98.70, "f1_px": 46.38, "ap_px": 44.72, "aupro": 92.66},
    {"category": "pipe_fryum", "auroc_sp": 98.34, "f1_sp": 96.04, "ap_sp": 99.16, "auroc_px": 99.40, "f1_px": 67.56, "ap_px": 67.90, "aupro": 97.32},
]

repository_mean = {
    "category": "mean",
    "auroc_sp": 92.57,
    "f1_sp": 89.06,
    "ap_sp": 93.31,
    "auroc_px": 98.71,
    "f1_px": 48.90,
    "ap_px": 45.38,
    "aupro": 92.43,
}

reference = pd.DataFrame(reference_records + [repository_mean])
comparison = reproduced_with_mean[["category", *METRIC_COLUMNS]].merge(
    reference[["category", *METRIC_COLUMNS]],
    on="category",
    suffixes=("_reproduced", "_reference"),
    validate="one_to_one",
)

for metric in METRIC_COLUMNS:
    comparison[f"{metric}_delta"] = (
        comparison[f"{metric}_reproduced"] - comparison[f"{metric}_reference"]
    )

delta_columns = [f"{metric}_delta" for metric in METRIC_COLUMNS]
category_rows = comparison[comparison["category"] != "mean"]
max_absolute_delta = category_rows[delta_columns].abs().to_numpy().max()

alerts = []
for _, row in category_rows.iterrows():
    for metric in METRIC_COLUMNS:
        delta = float(row[f"{metric}_delta"])
        if abs(delta) > 0.10:
            alerts.append(
                {
                    "category": row["category"],
                    "metric": metric,
                    "reproduced": row[f"{metric}_reproduced"],
                    "reference": row[f"{metric}_reference"],
                    "delta": delta,
                }
            )

comparison.to_csv(RUN_ROOT / "visa_reference_comparison.csv", index=False)
comparison.to_excel(RUN_ROOT / "visa_reference_comparison.xlsx", index=False)
pd.DataFrame(alerts).to_csv(RUN_ROOT / "comparison-alerts.csv", index=False)

display(comparison[comparison["category"] == "mean"])
print(f"Sai khác tuyệt đối lớn nhất ở category rows: {max_absolute_delta:.6f}")
print(f"Số cảnh báo vượt 0.10 điểm phần trăm: {len(alerts)}")

summary_lines = [
    "# E02 VisA Baseline Summary",
    "",
    f"- Run ID: `{RUN_ID}`",
    f"- Completed categories: {len(records)}/12",
    f"- Maximum absolute category-metric delta: {max_absolute_delta:.6f} percentage points",
    f"- Deltas above 0.10 percentage points: {len(alerts)}",
    "",
    "## Reproduced Mean",
    "",
]

mean_record = reproduced_with_mean[reproduced_with_mean["category"] == "mean"].iloc[0]
for metric in METRIC_COLUMNS:
    summary_lines.append(f"- {metric}: {mean_record[metric]:.6f}")

(RUN_ROOT / "E02_SUMMARY.md").write_text(
    "\n".join(summary_lines) + "\n", encoding="utf-8"
)

completion = {
    "run_id": RUN_ID,
    "aggregated_at_utc": datetime.now(timezone.utc).isoformat(),
    "completed_categories": len(records),
    "max_absolute_category_metric_delta": float(max_absolute_delta),
    "alerts_over_0_10_percentage_points": len(alerts),
}
(RUN_ROOT / "E02_AGGREGATED.json").write_text(
    json.dumps(completion, indent=2) + "\n", encoding="utf-8"
)

if ACTIVE_RUN_FILE.is_file() and ACTIVE_RUN_FILE.read_text(encoding="utf-8").strip() == RUN_ID:
    ACTIVE_RUN_FILE.unlink()
```

Ngưỡng 0.10 điểm phần trăm chỉ tạo cảnh báo để kiểm tra. Nó không tự động quyết định E02 đạt hay không đạt.

## 15. Cell E02-12 - Đóng gói kết quả

```python
import shutil

EXPORT_ROOT = PROJECT_ROOT / "exports"
EXPORT_ROOT.mkdir(parents=True, exist_ok=True)
archive_path = shutil.make_archive(
    str(EXPORT_ROOT / RUN_ID),
    "zip",
    root_dir=RUN_ROOT,
)

print("Đã tạo ZIP:", archive_path)
print("Dung lượng ZIP:", Path(archive_path).stat().st_size / 2**20, "MiB")
```

Tải ZIP trong:

```text
MyDrive/CS2203_MuSc/exports/<E02_RUN_ID>.zip
```

Gửi ZIP để kiểm tra trước khi chạy E02Q định tính.

## 16. Tiêu chí hoàn thành E02

- [ ] Dataset đến từ URL Amazon S3 chính thức và SHA-256 được lưu.
- [ ] `meta.json` được tạo bằng script của đúng MuSc commit.
- [ ] Có `SUCCESS.json` cho đủ 12 category.
- [ ] Mỗi category có `command.sh`, `console.log`, `metrics.json` và `results.xlsx`.
- [ ] Baseline giữ `divide_num = 1`, batch size 4, image size 518 và đúng backbone.
- [ ] Có bảng per-category và mean.
- [ ] Có bảng so sánh với repository MuSc.
- [ ] Mọi cảnh báo trên 0.10 điểm phần trăm được giữ lại để phân tích.
- [ ] ZIP được lưu trên Google Drive và tải về thành công.

## 17. Khi gặp lỗi

### Colab không cấp GPU

Dừng tại E02-2 và thử lại khi quota GPU được phục hồi. Không chạy MuSc bằng CPU.

### Runtime hết RAM hoặc bị ngắt

Mở runtime mới, chạy lại E02-1 đến E02-7, sau đó E02-8R và E02-9. Kết quả category đã thành công không bị chạy lại.

### Archive bị lỗi

Cell E02-3 giữ bản cache lỗi bằng hậu tố `.invalid-<timestamp>` và tải lại từ nguồn chính thức. Không dùng archive nếu `tar -tf` thất bại.

### Một category báo CUDA out-of-memory

Không đổi `divide_num` ngay. Gửi `console.log`, `FAILED.json`, category và `environment.txt` để kiểm tra. `divide_num > 1` chỉ được dùng như một experiment khác vì nó thay đổi mutual-scoring population.
