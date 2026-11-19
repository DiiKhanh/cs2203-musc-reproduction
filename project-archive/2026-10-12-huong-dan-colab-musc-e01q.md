# Hướng dẫn Colab E01Q - Xuất và chọn trường hợp định tính MVTec AD

## 1. Mục tiêu

E01 đã hoàn thành bảng định lượng cho 15 category nhưng chạy với `vis=False`. E01Q chạy lại ba category có chủ đích với `vis=True`, sau đó tạo các panel gồm:

1. Ảnh gốc.
2. Ground-truth mask.
3. MuSc anomaly heatmap.
4. Heatmap chồng lên ảnh gốc, có đường viền ground truth.

Ba category được chọn:

| Category | Loại | Lý do chọn |
|---|---|---|
| `leather` | Texture | Kết quả segmentation mạnh, pixel AUROC 99.7170 và PRO 98.7345 |
| `transistor` | Object | PRO thấp nhất trong E01, 77.2087 |
| `screw` | Object | Image AUROC thấp nhất trong E01, 82.3119 |

E01Q không thay đổi baseline. Nó giữ backbone, image size, feature layers, LNAMD degrees, `divide_num=1`, batch size và seed của E01.

## 2. Kết quả cần có

- Heatmap `whole_norm` cho toàn bộ test image của ba category.
- Bảng điểm định tính cho từng ảnh.
- Sáu trường hợp được chọn cho mỗi category:
  - Hai anomaly có pixel AUROC cao nhất.
  - Hai anomaly có pixel AUROC thấp nhất.
  - Hai normal image có top-1-percent heatmap score cao nhất.
- Các asset riêng và panel dùng cho báo cáo.
- Một ZIP nhỏ không chứa toàn bộ heatmap trung gian.

Các normal image được chọn chỉ là **false-positive candidates** theo cường độ heatmap. Không gọi chúng là false positive chính thức nếu chưa xác định classification threshold.

## 3. Thời gian dự kiến

Trên Tesla T4, ba category cần khoảng 30 đến 40 phút inference, chưa tính thời gian ghi heatmap lên Google Drive.

## 4. Các cell Colab

### Cell Q0 - Tạo hoặc resume E01Q Run ID

Chạy cell này dù E01 runtime hiện tại vẫn còn hoạt động.

```python
from google.colab import drive
drive.mount("/content/drive", force_remount=False)

from datetime import datetime, timezone
from pathlib import Path
import json
import os

UPSTREAM_COMMIT = "b76b93da8bd3096a99964a96ae29d46f197a0651"
PROJECT_ROOT = Path("/content/drive/MyDrive/CS2203_MuSc")
RUNS_ROOT = PROJECT_ROOT / "runs"
EXPORT_ROOT = PROJECT_ROOT / "exports"
SOURCE_E01_RUN_ID = "20261012_074856_E01_mvtec_all_clip_l14_336_baseline"
SOURCE_E01_ROOT = RUNS_ROOT / SOURCE_E01_RUN_ID
ACTIVE_Q_RUN_FILE = PROJECT_ROOT / "active-e01q-run.txt"

CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill", "screw",
    "tile", "toothbrush", "transistor", "wood", "zipper",
]
QUAL_CATEGORIES = ["leather", "transistor", "screw"]

RUNS_ROOT.mkdir(parents=True, exist_ok=True)
EXPORT_ROOT.mkdir(parents=True, exist_ok=True)

if ACTIVE_Q_RUN_FILE.is_file():
    candidate_id = ACTIVE_Q_RUN_FILE.read_text(encoding="utf-8").strip()
    candidate_root = RUNS_ROOT / candidate_id
    if candidate_root.is_dir() and not (candidate_root / "E01Q_DONE.json").exists():
        RUN_ID = candidate_id
        print("Tiếp tục E01Q Run ID:", RUN_ID)
    else:
        ACTIVE_Q_RUN_FILE.unlink()
        RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S_E01Q_mvtec_qualitative")
        ACTIVE_Q_RUN_FILE.write_text(RUN_ID + "\n", encoding="utf-8")
        print("Tạo E01Q Run ID mới:", RUN_ID)
else:
    RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S_E01Q_mvtec_qualitative")
    ACTIVE_Q_RUN_FILE.write_text(RUN_ID + "\n", encoding="utf-8")
    print("Tạo E01Q Run ID mới:", RUN_ID)

RUN_ROOT = RUNS_ROOT / RUN_ID
CATEGORY_ROOT = RUN_ROOT / "categories"
SESSION_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
SESSION_ROOT = RUN_ROOT / "sessions" / SESSION_ID

CATEGORY_ROOT.mkdir(parents=True, exist_ok=True)
SESSION_ROOT.mkdir(parents=True, exist_ok=False)

assert SOURCE_E01_ROOT.is_dir(), f"Không tìm thấy E01 nguồn: {SOURCE_E01_ROOT}"
assert (SOURCE_E01_ROOT / "E01_AGGREGATED.json").is_file()

print("E01Q run root:", RUN_ROOT)
print("E01 source root:", SOURCE_E01_ROOT)
```

### Cell Q1 - Kiểm tra runtime hiện tại

```python
from pathlib import Path

if "DATA_ROOT" not in globals():
    DATA_ROOT = Path("/content/data/mvtec_anomaly_detection")
    if not DATA_ROOT.is_dir():
        DATA_ROOT = Path("/content/data")

if "REPO_ROOT" not in globals():
    REPO_ROOT = Path("/content/MuSc")

dataset_ready = (
    DATA_ROOT.is_dir()
    and all((DATA_ROOT / category / "test").is_dir() for category in QUAL_CATEGORIES)
)
repo_ready = (REPO_ROOT / "examples" / "musc_main.py").is_file()
memory_patch_ready = (
    repo_ready
    and "gc.collect()" in (REPO_ROOT / "models" / "musc.py").read_text(
        encoding="utf-8", errors="replace"
    )
)

print("Dataset ready:", dataset_ready, DATA_ROOT)
print("Repository ready:", repo_ready, REPO_ROOT)
print("CPU memory patch ready:", memory_patch_ready)

if not (dataset_ready and repo_ready and memory_patch_ready):
    raise RuntimeError(
        "Runtime E01 không còn đầy đủ. Thực hiện mục 5 bên dưới, "
        "sau đó chạy lại Cell Q1."
    )
```

Nếu ba dòng đều là `True`, bỏ qua mục 5 và tiếp tục Cell Q2.

### Cell Q2 - Lưu cấu hình và định nghĩa hàm chạy qualitative category

```python
from datetime import datetime, timezone
import json
import pandas as pd
import shlex
import subprocess
import sys
import time

METRIC_COLUMNS = [
    "auroc_px", "f1_px", "ap_px", "aupro",
    "auroc_sp", "f1_sp", "ap_sp",
]

QUAL_CONFIG = {
    "source_e01_run_id": SOURCE_E01_RUN_ID,
    "categories": QUAL_CATEGORIES,
    "dataset_name": "mvtec_ad",
    "backbone_name": "ViT-L-14-336",
    "pretrained": "openai",
    "feature_layers": [5, 11, 17, 23],
    "img_resize": 518,
    "divide_num": 1,
    "r_list": [1, 3, 5],
    "batch_size": 4,
    "vis": True,
    "vis_type": "whole_norm",
    "save_excel": True,
    "seed_in_upstream_entrypoint": 42,
}

(RUN_ROOT / "qualitative-config.json").write_text(
    json.dumps(QUAL_CONFIG, indent=2) + "\n",
    encoding="utf-8",
)

source_metrics = pd.read_csv(SOURCE_E01_ROOT / "mvtec_ad_reproduced.csv")

def successful_qual_attempt(category):
    success_files = sorted((CATEGORY_ROOT / category).glob("attempt_*/SUCCESS.json"))
    return success_files[-1].parent if success_files else None

def run_qual_category(category):
    previous = successful_qual_attempt(category)
    if previous is not None:
        metrics = json.loads((previous / "metrics.json").read_text(encoding="utf-8"))
        print(f"SKIP {category}: đã thành công tại {previous}")
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
        "--dataset_name", "mvtec_ad",
        "--class_name", category,
        "--backbone_name", "ViT-L-14-336",
        "--pretrained", "openai",
        "--feature_layers", "5", "11", "17", "23",
        "--img_resize", "518",
        "--divide_num", "1",
        "--r_list", "1", "3", "5",
        "--batch_size", "4",
        "--output_dir", str(output_root),
        "--vis", "True",
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
    status = {
        "category": category,
        "attempt": attempt_number,
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": elapsed_seconds,
        "return_code": return_code,
    }

    if return_code != 0:
        (attempt_root / "FAILED.json").write_text(
            json.dumps(status, indent=2) + "\n", encoding="utf-8"
        )
        raise RuntimeError(f"{category} thất bại. Xem {log_path}")

    result_root = (
        output_root / "mvtec_ad" / "ViT-L-14-336" / "imagesize518"
    )
    result_file = result_root / "results.xlsx"
    heatmap_root = result_root / category
    assert result_file.is_file(), f"Thiếu {result_file}"
    assert heatmap_root.is_dir(), f"Thiếu {heatmap_root}"

    heatmaps = [
        path for path in heatmap_root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}
    ]
    expected_images = list((DATA_ROOT / category / "test").glob("*/*"))
    assert len(heatmaps) == len(expected_images), (
        f"Số heatmap không khớp: {len(heatmaps)} != {len(expected_images)}"
    )

    frame = pd.read_excel(result_file)
    frame = frame.rename(columns={frame.columns[0]: "category"})
    row = frame.iloc[0]
    assert row["category"] == category

    metrics = {"category": category}
    metrics.update({column: float(row[column]) for column in METRIC_COLUMNS})
    metrics["elapsed_seconds"] = elapsed_seconds
    metrics["heatmap_count"] = len(heatmaps)
    metrics["attempt_root"] = str(attempt_root)

    source_row = source_metrics[source_metrics["category"] == category].iloc[0]
    deltas = {
        column: metrics[column] - float(source_row[column])
        for column in METRIC_COLUMNS
    }
    max_delta = max(abs(value) for value in deltas.values())
    metrics["max_absolute_delta_from_e01"] = max_delta
    assert max_delta <= 0.05, (
        f"{category} lệch E01 quá 0.05 điểm phần trăm: {max_delta}"
    )

    (attempt_root / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    (attempt_root / "SUCCESS.json").write_text(
        json.dumps(status, indent=2) + "\n", encoding="utf-8"
    )
    print(f"DONE {category}: {elapsed_seconds / 60:.2f} phút, {len(heatmaps)} heatmap")
    print("Max delta from E01:", max_delta)
    return metrics
```

### Cell Q3 - Chạy hoặc resume ba category

```python
qual_metrics = {}
for index, category in enumerate(QUAL_CATEGORIES, start=1):
    print(f"\n===== [{index}/{len(QUAL_CATEGORIES)}] {category} =====")
    qual_metrics[category] = run_qual_category(category)

print("Đã hoàn thành inference định tính cho ba category.")
```

Nếu runtime bị ngắt, chạy lại Q0, hoàn tất setup nếu cần, rồi chạy Q2 và Q3. Category có `SUCCESS.json` sẽ được bỏ qua.

### Cell Q4 - Khôi phục giá trị heatmap và xếp hạng case

MuSc lưu heatmap bằng OpenCV JET với 256 màu. Vì PNG là lossless, cell này đảo bảng màu về thang điểm `0..255`, sau đó tính pixel AUROC cho từng anomaly image.

```python
import cv2
import numpy as np
from sklearn.metrics import roc_auc_score

jet_lut = cv2.applyColorMap(
    np.arange(256, dtype=np.uint8).reshape(256, 1),
    cv2.COLORMAP_JET,
).reshape(256, 3)

lut_keys = (
    jet_lut[:, 0].astype(np.uint32)
    | (jet_lut[:, 1].astype(np.uint32) << 8)
    | (jet_lut[:, 2].astype(np.uint32) << 16)
)
inverse_lut = np.full(1 << 24, -1, dtype=np.int16)
inverse_lut[lut_keys] = np.arange(256, dtype=np.int16)

def heatmap_to_score(path):
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    assert bgr is not None, f"Không đọc được heatmap: {path}"
    keys = (
        bgr[:, :, 0].astype(np.uint32)
        | (bgr[:, :, 1].astype(np.uint32) << 8)
        | (bgr[:, :, 2].astype(np.uint32) << 16)
    )
    score = inverse_lut[keys]
    assert np.all(score >= 0), f"Heatmap chứa màu ngoài JET LUT: {path}"
    return score.astype(np.float32) / 255.0

records = []
for category in QUAL_CATEGORIES:
    attempt_root = successful_qual_attempt(category)
    assert attempt_root is not None
    heatmap_root = (
        attempt_root / "output" / "mvtec_ad" / "ViT-L-14-336" /
        "imagesize518" / category
    )

    heatmaps = sorted(
        path for path in heatmap_root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )
    for heatmap_path in heatmaps:
        anomaly_type = heatmap_path.parent.name
        image_name = heatmap_path.name
        image_path = DATA_ROOT / category / "test" / anomaly_type / image_name
        assert image_path.is_file(), f"Thiếu ảnh gốc: {image_path}"

        score = heatmap_to_score(heatmap_path)
        flat = score.reshape(-1)
        top_count = max(1, int(np.ceil(flat.size * 0.01)))
        top1_mean = float(np.partition(flat, -top_count)[-top_count:].mean())

        mask_path = ""
        pixel_auroc = np.nan
        if anomaly_type != "good":
            mask_candidates = sorted(
                (DATA_ROOT / category / "ground_truth" / anomaly_type).glob(
                    f"{Path(image_name).stem}*"
                )
            )
            assert len(mask_candidates) == 1, (
                f"Không tìm thấy duy nhất mask cho {image_path}: {mask_candidates}"
            )
            mask_path = str(mask_candidates[0])
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (score.shape[1], score.shape[0]), interpolation=cv2.INTER_NEAREST)
            target = (mask > 0).astype(np.uint8).reshape(-1)
            assert target.min() == 0 and target.max() == 1
            pixel_auroc = float(roc_auc_score(target, flat))

        records.append(
            {
                "category": category,
                "anomaly_type": anomaly_type,
                "image_name": image_name,
                "image_path": str(image_path),
                "mask_path": mask_path,
                "heatmap_path": str(heatmap_path),
                "pixel_auroc_from_saved_heatmap": pixel_auroc,
                "top1_percent_heatmap_mean": top1_mean,
            }
        )

case_scores = pd.DataFrame(records)
case_scores.to_csv(RUN_ROOT / "qualitative-case-scores.csv", index=False)

selected_parts = []
for category in QUAL_CATEGORIES:
    category_rows = case_scores[case_scores["category"] == category]
    anomaly_rows = category_rows[category_rows["anomaly_type"] != "good"]
    normal_rows = category_rows[category_rows["anomaly_type"] == "good"]

    best = anomaly_rows.nlargest(2, "pixel_auroc_from_saved_heatmap").copy()
    best["selection_reason"] = "best_localization"
    worst = anomaly_rows.nsmallest(2, "pixel_auroc_from_saved_heatmap").copy()
    worst["selection_reason"] = "worst_localization"
    normal_fp = normal_rows.nlargest(2, "top1_percent_heatmap_mean").copy()
    normal_fp["selection_reason"] = "high_response_normal_candidate"

    selected_parts.extend([best, worst, normal_fp])

selected = pd.concat(selected_parts, ignore_index=True)
selected = selected.drop_duplicates(
    subset=["category", "anomaly_type", "image_name"], keep="first"
)
selected.to_csv(RUN_ROOT / "selected-qualitative-cases.csv", index=False)

display(
    selected[
        [
            "category", "anomaly_type", "image_name", "selection_reason",
            "pixel_auroc_from_saved_heatmap", "top1_percent_heatmap_mean",
        ]
    ]
)
```

Giá trị `pixel_auroc_from_saved_heatmap` được tính từ heatmap 8-bit phục vụ chọn case, không thay thế pixel AUROC chính thức được tính từ raw anomaly map trong E01.

### Cell Q5 - Tạo asset và panel cho báo cáo

```python
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

ASSET_ROOT = RUN_ROOT / "selected-assets"
ASSET_ROOT.mkdir(parents=True, exist_ok=True)

def prepare_original(path):
    image = Image.open(path).convert("RGB")
    image = TF.resize(
        image,
        518,
        interpolation=InterpolationMode.BICUBIC,
        antialias=True,
    )
    image = TF.center_crop(image, [518, 518])
    return np.asarray(image)

asset_records = []
for _, row in selected.iterrows():
    key = f"{row['category']}__{row['anomaly_type']}__{Path(row['image_name']).stem}"
    case_root = ASSET_ROOT / key
    case_root.mkdir(parents=True, exist_ok=True)

    original = prepare_original(row["image_path"])
    heatmap_bgr = cv2.imread(row["heatmap_path"], cv2.IMREAD_COLOR)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    if heatmap_rgb.shape[:2] != original.shape[:2]:
        heatmap_rgb = cv2.resize(heatmap_rgb, (original.shape[1], original.shape[0]))

    if row["mask_path"]:
        mask = cv2.imread(row["mask_path"], cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (original.shape[1], original.shape[0]), interpolation=cv2.INTER_NEAREST)
    else:
        mask = np.zeros(original.shape[:2], dtype=np.uint8)

    overlay = cv2.addWeighted(original, 0.55, heatmap_rgb, 0.45, 0)
    contours, _ = cv2.findContours(
        (mask > 0).astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    cv2.drawContours(overlay, contours, -1, (255, 255, 255), 2)

    Image.fromarray(original).save(case_root / "original.png")
    Image.fromarray(mask).save(case_root / "ground_truth.png")
    Image.fromarray(heatmap_rgb).save(case_root / "heatmap.png")
    Image.fromarray(overlay).save(case_root / "overlay.png")

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.4))
    axes[0].imshow(original)
    axes[0].set_title("Original")
    axes[1].imshow(mask, cmap="gray", vmin=0, vmax=255)
    axes[1].set_title("Ground truth")
    axes[2].imshow(heatmap_rgb)
    axes[2].set_title("MuSc heatmap")
    axes[3].imshow(overlay)
    axes[3].set_title("Overlay + GT contour")
    for axis in axes:
        axis.axis("off")

    px_auc = row["pixel_auroc_from_saved_heatmap"]
    px_text = "N/A" if pd.isna(px_auc) else f"{px_auc:.4f}"
    fig.suptitle(
        f"{row['category']} | {row['anomaly_type']} | {row['image_name']} | "
        f"{row['selection_reason']} | saved-map pixel AUROC={px_text}",
        fontsize=11,
    )
    fig.tight_layout()
    panel_path = case_root / "panel.png"
    fig.savefig(panel_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    asset_records.append(
        {
            **row.to_dict(),
            "asset_root": str(case_root),
            "panel_path": str(panel_path),
        }
    )

asset_index = pd.DataFrame(asset_records)
asset_index.to_csv(RUN_ROOT / "selected-assets-index.csv", index=False)
print("Đã tạo", len(asset_index), "panel tại", ASSET_ROOT)
```

### Cell Q6 - Xem nhanh các panel

```python
from IPython.display import display

for category in QUAL_CATEGORIES:
    print(f"\n===== {category} =====")
    rows = asset_index[asset_index["category"] == category]
    for _, row in rows.iterrows():
        print(
            row["selection_reason"],
            row["anomaly_type"],
            row["image_name"],
        )
        display(Image.open(row["panel_path"]))
```

Khi xem panel, ghi nhận:

- Heatmap có phủ đúng vùng ground truth hay không.
- Heatmap có lan rộng ra background hay chi tiết bình thường hay không.
- Defect nhỏ có bị bỏ sót hay không.
- Normal image có vùng phản hồi mạnh tại biên, texture hoặc vật thể phụ hay không.

### Cell Q7 - Hoàn tất và tạo ZIP nhỏ

ZIP này giữ selected assets, bảng lựa chọn, config, metrics, command và log. Nó bỏ toàn bộ thư mục `output` chứa hàng trăm heatmap trung gian.

```python
import zipfile

done = {
    "run_id": RUN_ID,
    "source_e01_run_id": SOURCE_E01_RUN_ID,
    "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    "categories": QUAL_CATEGORIES,
    "selected_case_count": int(len(asset_index)),
    "status": "completed",
}
(RUN_ROOT / "E01Q_DONE.json").write_text(
    json.dumps(done, indent=2) + "\n", encoding="utf-8"
)

zip_path = EXPORT_ROOT / f"{RUN_ID}.zip"
with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
    for path in RUN_ROOT.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(RUN_ROOT)
        if "output" in relative.parts:
            continue
        archive.write(path, arcname=str(relative))

assert zip_path.is_file()
print("E01Q hoàn tất.")
print("ZIP:", zip_path)
print("Dung lượng ZIP:", zip_path.stat().st_size / 2**20, "MiB")
```

Tải ZIP trong `MyDrive/CS2203_MuSc/exports/` và gửi để kiểm tra. Không xóa thư mục E01Q trên Drive vì toàn bộ heatmap trung gian vẫn nằm ở đó.

## 5. Khôi phục khi runtime E01 đã mất

Nếu Cell Q1 báo `False`, giữ nguyên E01Q Run ID vừa tạo và thực hiện:

1. Trong [E01 runbook](2026-10-11-huong-dan-colab-musc-e01.md), chạy **Cell 2** để cài dependency.
2. Chạy **Cell 3** phiên bản mới để tải và giải nén MVTec AD trong `/content`.
3. Chạy **Cell 4** để clone đúng MuSc commit và sửa import.
4. Chạy **Cell 4M** để áp dụng bản vá CPU RAM.
5. Chạy **Cell 5** để khôi phục CLIP weights.
6. Quay lại guide này và chạy lại Cell Q1, sau đó tiếp tục từ Q2.

Không chạy Cell 1 của E01 vì E01 đã aggregated xong. Cell Q0 của guide này đã tạo đúng `RUN_ROOT` và `SESSION_ROOT` cho E01Q.

## 6. Tiêu chí chấp nhận E01Q

E01Q đạt khi:

- [ ] Ba category đều có `SUCCESS.json`.
- [ ] Số heatmap bằng số test image của từng category.
- [ ] Mỗi metric rerun lệch E01 không quá 0.05 điểm phần trăm.
- [ ] Có selected case từ cả strong và weak categories.
- [ ] Mỗi selected case có original, mask, heatmap, overlay và panel.
- [ ] Các case normal được gọi là candidates, không gán nhãn false positive khi chưa có threshold.
- [ ] ZIP nhỏ được tạo và tải về để kiểm tra.
