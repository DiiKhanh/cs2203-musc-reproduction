# MuSc Classroom Demo - Hướng dẫn thiết lập Google Colab

## Mục tiêu và phạm vi

Notebook này thiết lập app demo MuSc trên một Google Colab runtime mới có
NVIDIA T4 GPU. App có hai chế độ độc lập:

- **Prepared Benchmark** đọc metric và hình ảnh từ artifact đã chuẩn bị sẵn.
- **Live Inference** nhận một tập 5-200 ảnh chưa có nhãn thuộc cùng một
  category sản phẩm công nghiệp. Khoảng khuyến nghị là 40-200 ảnh.

Live Inference chỉ trả về anomaly score tương đối trong chính tập ảnh đã tải
lên, heatmap, overlay và danh sách xếp hạng. App không trả về nhãn lỗi chắc chắn,
benchmark metric cho tập mới hoặc kết luận y khoa.

## Chuẩn bị trước khi mở notebook

### 1. Chuẩn bị GitHub token

Repository reproduction đang ở chế độ private. Tạo một fine-grained personal
access token chỉ có quyền đọc repository, sau đó:

1. Mở Colab.
2. Chọn biểu tượng chìa khóa **Secrets** ở thanh bên trái.
3. Tạo secret tên chính xác là `GITHUB_TOKEN`.
4. Bật quyền cho notebook được đọc secret này.

Không dán token trực tiếp vào code cell, output hoặc Google Drive.

### 2. Upload Prepared Benchmark bundle

Upload file `musc-classroom-prepared-bundle.zip` vào vị trí sau trong Google
Drive:

```text
MyDrive/CS2203_MuSc/musc-classroom-prepared-bundle.zip
```

ZIP phải chứa `manifest.json` ngay ở thư mục gốc. Notebook sẽ giải nén bundle
vào `/content`, không chạy app trực tiếp trên Google Drive.

### 3. Chọn T4 GPU

Trong Colab, chọn **Runtime > Change runtime type > T4 GPU**. Sau đó chạy các
cell theo đúng thứ tự và giữ cell cuối cùng hoạt động trong lúc trình bày.

## Cell 1 - Kiểm tra T4 GPU và khai báo runtime cố định

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
    "Không tìm thấy CUDA GPU. Hãy chọn Runtime > Change runtime type > T4 GPU."
)

os.environ["MUSC_ROOT"] = str(UPSTREAM_ROOT)
os.environ["MUSC_EAGER_INIT"] = "0"
os.environ["MUSC_COMMIT"] = PINNED_MUSC_COMMIT

print("Python:", sys.version)
print("Nền tảng:", platform.platform())
print("PyTorch:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0))
subprocess.run(["nvidia-smi"], check=True)
```

Kết quả mong đợi: GPU hiển thị `Tesla T4` hoặc một CUDA GPU tương thích.

## Cell 2 - Clone private reproduction repository bằng Colab Secrets

Cell này dùng `GIT_ASKPASS` tạm thời. Token không xuất hiện trong clone URL,
notebook output hoặc file đã commit.

```python
from google.colab import userdata
import os
import shutil
import stat
import subprocess
import tempfile
from pathlib import Path

github_token = userdata.get("GITHUB_TOKEN")
assert github_token, (
    "Chưa có GITHUB_TOKEN. Hãy thêm token vào Colab Secrets và cấp quyền đọc."
)

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
print("Đã clone reproduction repository:", REPRO_ROOT)
```

## Cell 3 - Clone, pin và kiểm tra official MuSc source

Cell này kiểm tra chính xác upstream commit trước khi import model. Patch chỉ sửa
thứ tự Python import trên Colab, không thay đổi CLIP, LNAMD, MSM hoặc RsCIN.

```bash
set -euo pipefail
rm -rf /content/MuSc
git clone https://github.com/xrli-U/MuSc.git /content/MuSc
git -C /content/MuSc checkout --detach "$MUSC_COMMIT"
ACTUAL_COMMIT="$(git -C /content/MuSc rev-parse HEAD)"
test "$ACTUAL_COMMIT" = "b76b93da8bd3096a99964a96ae29d46f197a0651"
git -C /content/MuSc apply /content/cs2203-musc-reproduction/patches/colab-import-compatibility.patch
printf 'Đã kiểm tra MuSc commit: %s\n' "$ACTUAL_COMMIT"
```

## Cell 4 - Cài dependencies mà không thay PyTorch của Colab

`requirements-demo.txt` không cài lại PyTorch hoặc torchvision. Cell này kiểm
tra phiên bản PyTorch trước và sau khi cài dependencies.

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
print("Giữ nguyên PyTorch của Colab:", torch_after_install.__version__)
```

## Cell 5 - Ghi nhận phiên bản môi trường

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

## Cell 6 - Mount Drive và tự giải nén Prepared Benchmark bundle

Cell này mặc định bật Prepared Benchmark. Nếu chỉ muốn thử Live Inference, đặt
`USE_PREPARED_RESULTS = False`.

```python
from google.colab import drive
from pathlib import Path
from zipfile import ZipFile
import os
import shutil
import stat

USE_PREPARED_RESULTS = True
PREPARED_ZIP = Path(
    "/content/drive/MyDrive/CS2203_MuSc/musc-classroom-prepared-bundle.zip"
)
PREPARED_ROOT = Path("/content/musc-classroom-prepared")

if USE_PREPARED_RESULTS:
    drive.mount("/content/drive")
    assert PREPARED_ZIP.is_file(), f"Không tìm thấy bundle: {PREPARED_ZIP}"

    if PREPARED_ROOT.exists():
        shutil.rmtree(PREPARED_ROOT)
    PREPARED_ROOT.mkdir(parents=True)

    with ZipFile(PREPARED_ZIP) as archive:
        for member in archive.infolist():
            member_path = Path(member.filename)
            file_type = (member.external_attr >> 16) & 0o170000
            assert not member_path.is_absolute(), f"ZIP path không an toàn: {member.filename}"
            assert ".." not in member_path.parts, f"ZIP traversal: {member.filename}"
            assert file_type != stat.S_IFLNK, f"ZIP symlink không được phép: {member.filename}"
        archive.extractall(PREPARED_ROOT)

    manifest_path = PREPARED_ROOT / "manifest.json"
    assert manifest_path.is_file(), f"Bundle thiếu manifest.json: {manifest_path}"
    os.environ["MUSC_PREPARED_ROOT"] = str(PREPARED_ROOT)
    print("Prepared Benchmark đã sẵn sàng:", PREPARED_ROOT)
else:
    os.environ.pop("MUSC_PREPARED_ROOT", None)
    print("Prepared Benchmark đã tắt. Live Inference vẫn có thể chạy.")
```

## Cell 7 - Preflight commit, config và Prepared Catalog

```python
import json
import subprocess
import sys

resolved_commit = subprocess.check_output(
    ["git", "-C", str(UPSTREAM_ROOT), "rev-parse", "HEAD"],
    text=True,
).strip()
assert resolved_commit == PINNED_MUSC_COMMIT

configuration = json.loads(DEMO_CONFIG.read_text(encoding="utf-8"))
assert configuration["upstream_commit"] == PINNED_MUSC_COMMIT
assert configuration["rscin_neighbors"] == [1, 2, 3]

if "MUSC_PREPARED_ROOT" in os.environ:
    sys.path.insert(0, str(REPRO_ROOT))
    from musc_demo.prepared_results import PreparedCatalog

    prepared_root = Path(os.environ["MUSC_PREPARED_ROOT"])
    catalog = PreparedCatalog.load(prepared_root / "manifest.json", prepared_root)
    assert catalog.error is None, catalog.error
    assert catalog.datasets(), "Prepared Catalog không có dataset khả dụng"
    print("Prepared datasets:", catalog.datasets())
    print("Số category MVTec AD:", len(catalog.categories("mvtec_ad")))

print("Preflight hoàn tất. Cell tiếp theo sẽ mở app; model được khởi tạo trong app.")
```

## Cell 8 - Mở app và Gradio share URL

Cell này tiếp tục chạy trong suốt buổi trình bày. Dừng cell để đóng temporary
share URL. App mở trước để Prepared Benchmark dùng được ngay; bấm
**Khởi tạo MuSc model** trong app khi cần chạy Live Inference.

```python
import os
import sys

os.environ["MUSC_ROOT"] = str(UPSTREAM_ROOT)
os.environ["MUSC_EAGER_INIT"] = "0"

if str(REPRO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPRO_ROOT))

print("[1/4] Import MuSc demo...", flush=True)
from musc_demo.app import (
    build_app,
    default_services,
    demo_theme,
    theme_css_path,
)

print("[2/4] Đọc cấu hình và Prepared Benchmark...", flush=True)
services = default_services()

print("[3/4] Xây dựng giao diện...", flush=True)
demo = build_app(services)

print("[4/4] Khởi động Gradio và tạo share URL...", flush=True)
demo.launch(
    share=True,
    debug=True,
    show_error=True,
    css_paths=theme_css_path(),
    theme=demo_theme(),
)
```

## Kiểm tra trước khi trình bày

1. Mở share URL trong cửa sổ ẩn danh.
2. Kiểm tra environment card hiển thị GPU và MuSc model sẵn sàng.
3. Trong Prepared Benchmark, chọn `carpet`, `screw` hoặc `transistor`, sau đó
   chọn run E04Q để xem original, ground truth, heatmap và overlay.
4. Với Live Inference, chỉ tải một tập ảnh cùng category sản phẩm công nghiệp.
5. Giữ sẵn CSV, ZIP kết quả và Prepared Benchmark làm phương án dự phòng.

## Xử lý lỗi thường gặp

- **Không có GPU:** đổi runtime sang T4 GPU rồi chạy lại từ Cell 1.
- **Clone private repository thất bại:** kiểm tra `GITHUB_TOKEN`, quyền đọc
  repository và quyền notebook truy cập secret.
- **Không tìm thấy bundle:** kiểm tra đúng tên file và vị trí trong
  `MyDrive/CS2203_MuSc/`.
- **Model initialization thất bại:** giữ Prepared Benchmark mở và dùng kết quả
  chuẩn bị sẵn để trình bày.
- **CUDA out of memory:** giảm số ảnh của batch rồi thử lại một lần.
- **Share URL không mở được:** dừng Cell 8, chạy lại và kiểm tra URL mới bằng cửa
  sổ ẩn danh.
