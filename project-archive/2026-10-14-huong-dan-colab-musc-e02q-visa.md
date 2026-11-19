# Hướng dẫn Colab E02Q - Phân tích định tính MuSc trên VisA

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: plan
- Origin Date: 2026-10-14
- Verification Status: UNVERIFIED
- Version Label: code_plan_v1

## 1. Mục tiêu

E02 đã hoàn thành baseline định lượng cho toàn bộ 12 category VisA với `vis=False`. E02Q chạy lại bốn category có chủ đích với `vis=True`, lưu per-image score trước và sau RsCIN, rồi tạo panel để giải thích các kết quả định lượng.

| Category | Vai trò trong E02Q | Lý do chọn |
|---|---|---|
| `pcb2` | Image-level failure case chính | Image AUROC lệch -1.1900, Image F1-max lệch -2.5062 và Image AP lệch -0.8869 điểm phần trăm |
| `pcb3` | Image-level degradation case | Image AUROC lệch -0.5497 và Image F1-max lệch -0.9387 điểm phần trăm |
| `pcb4` | Contrast case | Image F1-max cao hơn reference 0.9876 điểm phần trăm |
| `fryum` | Pixel-level reference audit | E02 đạt Pixel AUROC 97.9309, trong khi hàng official ghi 99.43 nhưng mean official lại ngụ ý khoảng 97.89 |

E02Q không thay đổi baseline. Nó giữ nguyên backbone, image size, feature layers, LNAMD degrees, `divide_num=1`, batch size, seed và CPU-memory patch của E02. Thay đổi duy nhất về quan sát là:

1. Bật `vis=True` với `whole_norm`.
2. Thêm instrumentation sau khi MuSc đã tính xong anomaly map và RsCIN score để lưu per-image score.
3. Tính per-image pixel AUROC trực tiếp từ raw anomaly map chỉ để chọn case.

Instrumentation không thay đổi input của `compute_metrics`, không thay đổi score và không thay đổi thuật toán MuSc.

## 2. Câu hỏi phân tích

E02Q trả lời ba câu hỏi:

1. Các anomaly có RsCIN score thấp nhất mang đặc điểm gì?
2. Các normal image có RsCIN score cao nhất kích hoạt trên vùng nào?
3. Category có image-level deviation lớn vẫn giữ localization tốt hay không?

```mermaid
flowchart LR
    A[Four selected VisA categories] --> B[MuSc baseline with visualization]
    B --> C[Raw anomaly maps]
    B --> D[Initial image scores]
    D --> E[RsCIN scores]
    C --> F[Per-image localization scores]
    E --> G[Rule-based case selection]
    F --> G
    G --> H[24 report panels]
```

## 3. Kết quả cần có

- Heatmap `whole_norm` cho toàn bộ 752 test image thuộc bốn category.
- Một `per-image-scores.csv` cho mỗi category, gồm:
  - Ground-truth image label.
  - Initial maximum anomaly-map score.
  - Initial score sau min-max normalization.
  - RsCIN score cuối cùng.
  - Mức thay đổi do RsCIN trên thang normalized.
  - Per-image pixel AUROC từ raw anomaly map đối với anomaly image.
  - Mean của một phần trăm pixel có raw anomaly score cao nhất.
- Sáu case khác nhau cho mỗi category:
  - Hai anomaly có RsCIN score thấp nhất.
  - Hai normal image có RsCIN score cao nhất.
  - Một anomaly có pixel AUROC cao nhất trong các anomaly chưa được chọn.
  - Một anomaly có pixel AUROC thấp nhất trong các anomaly chưa được chọn.
- Tối đa 24 panel gồm original, ground-truth mask, heatmap và overlay.
- Một ZIP nhỏ chứa selected assets, score tables, config, metrics, commands, logs và patch, nhưng không chứa toàn bộ heatmap trung gian.

Các case được gọi là `false_negative_candidate` và `false_positive_candidate` vì chúng được chọn theo thứ hạng score. Không gọi chúng là lỗi phân loại chính thức nếu chưa áp dụng threshold.

## 4. Thời gian dự kiến

E02 đã cần khoảng 70 phút subprocess cho bốn category này trên Tesla T4. Khi bật visualization và ghi 752 heatmap lên Google Drive, nên dành khoảng 80 đến 110 phút, chưa tính setup hoặc thời gian tải lại dataset.

## 5. Các cell Colab

Có thể thêm một section `E02Q - VisA qualitative analysis` vào cuối notebook E02, hoặc tạo notebook mới tên gợi ý:

```text
CS2203_MuSc_E02Q_VisA_Qualitative.ipynb
```

Không chạy lại Cell E02-1. E02 đã aggregated xong và E02Q phải có Run ID riêng.

### Cell Q0 - Mount Drive và tạo hoặc resume E02Q Run ID

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
SOURCE_E02_RUN_ID = "20261012_171400_E02_visa_all_clip_l14_336_baseline"
SOURCE_E02_ROOT = RUNS_ROOT / SOURCE_E02_RUN_ID
ACTIVE_Q_RUN_FILE = PROJECT_ROOT / "active-e02q-run.txt"

CATEGORIES = [
    "candle", "capsules", "cashew", "chewinggum", "fryum",
    "macaroni1", "macaroni2", "pcb1", "pcb2", "pcb3",
    "pcb4", "pipe_fryum",
]
QUAL_CATEGORIES = ["pcb2", "pcb3", "pcb4", "fryum"]

RUNS_ROOT.mkdir(parents=True, exist_ok=True)
EXPORT_ROOT.mkdir(parents=True, exist_ok=True)

if ACTIVE_Q_RUN_FILE.is_file():
    candidate_id = ACTIVE_Q_RUN_FILE.read_text(encoding="utf-8").strip()
    candidate_root = RUNS_ROOT / candidate_id
    if candidate_root.is_dir() and not (candidate_root / "E02Q_DONE.json").exists():
        RUN_ID = candidate_id
        print("Tiếp tục E02Q Run ID:", RUN_ID)
    else:
        ACTIVE_Q_RUN_FILE.unlink()
        RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S_E02Q_visa_qualitative")
        ACTIVE_Q_RUN_FILE.write_text(RUN_ID + "\n", encoding="utf-8")
        print("Tạo E02Q Run ID mới:", RUN_ID)
else:
    RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S_E02Q_visa_qualitative")
    ACTIVE_Q_RUN_FILE.write_text(RUN_ID + "\n", encoding="utf-8")
    print("Tạo E02Q Run ID mới:", RUN_ID)

RUN_ROOT = RUNS_ROOT / RUN_ID
CATEGORY_ROOT = RUN_ROOT / "categories"
SESSION_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
SESSION_ROOT = RUN_ROOT / "sessions" / SESSION_ID

CATEGORY_ROOT.mkdir(parents=True, exist_ok=True)
SESSION_ROOT.mkdir(parents=True, exist_ok=False)

assert SOURCE_E02_ROOT.is_dir(), f"Không tìm thấy E02 nguồn: {SOURCE_E02_ROOT}"
assert (SOURCE_E02_ROOT / "E02_AGGREGATED.json").is_file()
assert (SOURCE_E02_ROOT / "visa_reproduced.csv").is_file()
assert (SOURCE_E02_ROOT / "dataset-source.txt").is_file()

(RUN_ROOT / "run-id.txt").write_text(RUN_ID + "\n", encoding="utf-8")
(RUN_ROOT / "source-e02-run-id.txt").write_text(
    SOURCE_E02_RUN_ID + "\n", encoding="utf-8"
)

print("E02Q run root:", RUN_ROOT)
print("E02 source root:", SOURCE_E02_ROOT)
```

### Cell Q1 - Kiểm tra runtime hiện tại

```python
import json
import subprocess
import torch

if "DATA_ROOT" not in globals():
    DATA_ROOT = Path("/content/visa-extract")

if "REPO_ROOT" not in globals():
    REPO_ROOT = Path("/content/MuSc")

meta_path = DATA_ROOT / "meta.json"
meta = None
if meta_path.is_file():
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

dataset_ready = (
    DATA_ROOT.is_dir()
    and meta is not None
    and "test" in meta
    and all(category in meta["test"] for category in QUAL_CATEGORIES)
    and all(
        (DATA_ROOT / sample["img_path"]).is_file()
        and (
            sample["mask_path"] == ""
            or (DATA_ROOT / sample["mask_path"]).is_file()
        )
        for category in QUAL_CATEGORIES
        for sample in meta["test"][category]
    )
)

repo_ready = (REPO_ROOT / "examples" / "musc_main.py").is_file()
actual_commit = ""
if repo_ready:
    actual_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        check=True,
    ).stdout.strip()

musc_path = REPO_ROOT / "models" / "musc.py"
memory_patch_ready = (
    musc_path.is_file()
    and "gc.collect()" in musc_path.read_text(encoding="utf-8", errors="replace")
)
gpu_ready = torch.cuda.is_available()

print("Dataset ready:", dataset_ready, DATA_ROOT)
print("Repository ready:", repo_ready, REPO_ROOT)
print("Commit:", actual_commit)
print("CPU memory patch ready:", memory_patch_ready)
print("GPU ready:", gpu_ready, torch.cuda.get_device_name(0) if gpu_ready else "N/A")

if not (
    dataset_ready
    and repo_ready
    and actual_commit == UPSTREAM_COMMIT
    and memory_patch_ready
    and gpu_ready
):
    raise RuntimeError(
        "Runtime chưa đủ điều kiện. Thực hiện mục 6 bên dưới, "
        "sau đó chạy lại Cell Q1."
    )
```

Nếu tất cả điều kiện đều đạt, tiếp tục Cell Q2. Nếu runtime E02 đã mất, thực hiện mục 6 trước.

### Cell Q2 - Thêm instrumentation lưu per-image score

Cell này thêm việc ghi CSV sau khi raw anomaly map, initial image score và RsCIN score đã được tính. Nó không thay đổi các giá trị được truyền vào `compute_metrics`.

```python
import subprocess
import sys

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

source = replace_once(
    source,
    "from utils.metrics import compute_metrics\n",
    "from utils.metrics import compute_metrics\n"
    "from sklearn.metrics import roc_auc_score\n",
    "per-image AUROC import",
)

old_metrics_block = (
    "        pr_sp = np.array(scores_cls)\n"
    "        gt_sp = np.array(gt_list)\n"
    "        gt_px = torch.cat(img_masks, dim=0).numpy().astype(np.int32)\n"
    "        pr_px = anomaly_maps\n"
    "        image_metric, pixel_metric = compute_metrics(gt_sp, pr_sp, gt_px, pr_px)\n"
)

new_metrics_block = (
    "        pr_sp = np.array(scores_cls)\n"
    "        gt_sp = np.array(gt_list)\n"
    "        gt_px = torch.cat(img_masks, dim=0).numpy().astype(np.int32)\n"
    "        pr_px = anomaly_maps\n"
    "\n"
    "        # Export observational per-image scores without changing metric inputs.\n"
    "        assert len(image_path_list) == len(gt_sp) == len(ac_score) == len(pr_sp)\n"
    "        initial_score_range = ac_score.max() - ac_score.min()\n"
    "        assert initial_score_range > 0\n"
    "        initial_normalized = (ac_score - ac_score.min()) / initial_score_range\n"
    "        per_image_records = []\n"
    "        for sample_index, image_path in enumerate(image_path_list):\n"
    "            flat_score = pr_px[sample_index].reshape(-1)\n"
    "            top_count = max(1, int(np.ceil(flat_score.size * 0.01)))\n"
    "            top1_mean = float(np.partition(flat_score, -top_count)[-top_count:].mean())\n"
    "            pixel_auroc = ''\n"
    "            if int(gt_sp[sample_index]) == 1:\n"
    "                flat_mask = gt_px[sample_index].reshape(-1)\n"
    "                assert flat_mask.min() == 0 and flat_mask.max() == 1\n"
    "                pixel_auroc = float(roc_auc_score(flat_mask, flat_score))\n"
    "            per_image_records.append(\n"
    "                {\n"
    "                    'category': category,\n"
    "                    'image_path': image_path,\n"
    "                    'is_anomaly': int(gt_sp[sample_index]),\n"
    "                    'initial_max_score_raw': float(ac_score[sample_index]),\n"
    "                    'initial_normalized_score': float(initial_normalized[sample_index]),\n"
    "                    'rscin_score': float(pr_sp[sample_index]),\n"
    "                    'rscin_delta_from_initial_normalized': float(\n"
    "                        pr_sp[sample_index] - initial_normalized[sample_index]\n"
    "                    ),\n"
    "                    'pixel_auroc_raw': pixel_auroc,\n"
    "                    'top1_percent_raw_mean': top1_mean,\n"
    "                }\n"
    "            )\n"
    "        per_image_score_path = os.path.join(\n"
    "            self.output_dir, f'{category}_per-image-scores.csv'\n"
    "        )\n"
    "        import pandas as pd\n"
    "        pd.DataFrame(per_image_records).to_csv(per_image_score_path, index=False)\n"
    "\n"
    "        image_metric, pixel_metric = compute_metrics(gt_sp, pr_sp, gt_px, pr_px)\n"
)

source = replace_once(
    source,
    old_metrics_block,
    new_metrics_block,
    "per-image score export",
)
musc_path.write_bytes(source)

subprocess.run(
    [sys.executable, "-m", "py_compile", str(musc_path)],
    cwd=REPO_ROOT,
    check=True,
)

instrumentation_diff = subprocess.run(
    ["git", "diff", "--", "models/musc.py"],
    cwd=REPO_ROOT,
    text=True,
    stdout=subprocess.PIPE,
    check=True,
).stdout

assert "per_image_score_path" in instrumentation_diff
assert "pixel_auroc_raw" in instrumentation_diff
assert "compute_metrics(gt_sp, pr_sp, gt_px, pr_px)" in instrumentation_diff

(SESSION_ROOT / "musc-instrumentation-and-memory.diff").write_text(
    instrumentation_diff,
    encoding="utf-8",
)
print(instrumentation_diff)
```

Cell phải in ra diff chứa CPU-memory patch và instrumentation. Nếu marker không khớp, dừng lại, không sửa thủ công theo phỏng đoán.

### Cell Q3 - Lưu cấu hình và định nghĩa hàm chạy một category

```python
from datetime import datetime, timezone
import json
import math
import pandas as pd
import shlex
import shutil
import subprocess
import sys
import time

METRIC_COLUMNS = [
    "auroc_px", "f1_px", "ap_px", "aupro",
    "auroc_sp", "f1_sp", "ap_sp",
]

SCORE_COLUMNS = [
    "category",
    "image_path",
    "is_anomaly",
    "initial_max_score_raw",
    "initial_normalized_score",
    "rscin_score",
    "rscin_delta_from_initial_normalized",
    "pixel_auroc_raw",
    "top1_percent_raw_mean",
]

QUAL_CONFIG = {
    "source_e02_run_id": SOURCE_E02_RUN_ID,
    "categories": QUAL_CATEGORIES,
    "dataset_name": "visa",
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
    "metric_tolerance_percentage_points": 0.05,
    "selection_count_per_category": 6,
}

config_path = RUN_ROOT / "qualitative-config.json"
config_text = json.dumps(QUAL_CONFIG, indent=2, sort_keys=True) + "\n"
if config_path.exists():
    assert config_path.read_text(encoding="utf-8") == config_text
else:
    config_path.write_text(config_text, encoding="utf-8")

source_metrics = pd.read_csv(SOURCE_E02_ROOT / "visa_reproduced.csv")
source_counts = json.loads(
    (SOURCE_E02_ROOT / "visa-test-counts.json").read_text(encoding="utf-8")
)
assert set(QUAL_CATEGORIES).issubset(source_counts)

def successful_qual_attempt(category):
    success_files = sorted((CATEGORY_ROOT / category).glob("attempt_*/SUCCESS.json"))
    if not success_files:
        return None
    attempt_root = success_files[-1].parent
    assert (attempt_root / "metrics.json").is_file()
    assert (attempt_root / "per-image-scores.csv").is_file()
    return attempt_root

def run_qual_category(category):
    assert category in QUAL_CATEGORIES, f"Category không hợp lệ: {category}"

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
        "gpu": torch.cuda.get_device_name(0),
    }

    if return_code != 0:
        (attempt_root / "FAILED.json").write_text(
            json.dumps(status, indent=2) + "\n", encoding="utf-8"
        )
        raise RuntimeError(f"{category} thất bại. Xem {log_path}")

    try:
        result_root = (
            output_root / "visa" / "ViT-L-14-336" / "imagesize518"
        )
        result_file = result_root / "results.xlsx"
        heatmap_root = result_root / category
        score_file = result_root / f"{category}_per-image-scores.csv"

        assert result_file.is_file(), f"Thiếu {result_file}"
        assert heatmap_root.is_dir(), f"Thiếu {heatmap_root}"
        assert score_file.is_file(), f"Thiếu {score_file}"

        heatmaps = [
            path for path in heatmap_root.rglob("*")
            if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}
        ]
        expected_count = int(source_counts[category])
        assert len(heatmaps) == expected_count, (
            f"Số heatmap không khớp: {len(heatmaps)} != {expected_count}"
        )

        scores = pd.read_csv(score_file)
        assert list(scores.columns) == SCORE_COLUMNS
        assert len(scores) == expected_count
        assert scores["image_path"].is_unique
        assert set(scores["category"]) == {category}
        assert set(scores["is_anomaly"].astype(int)).issubset({0, 1})
        assert scores["rscin_score"].map(math.isfinite).all()
        assert scores["initial_normalized_score"].between(0, 1).all()
        assert all(Path(path).is_file() for path in scores["image_path"])

        frame = pd.read_excel(result_file)
        frame = frame.rename(columns={frame.columns[0]: "category"})
        row = frame.iloc[0]
        assert row["category"] == category

        metrics = {"category": category}
        metrics.update({column: float(row[column]) for column in METRIC_COLUMNS})
        for column in METRIC_COLUMNS:
            assert math.isfinite(metrics[column]) and 0 <= metrics[column] <= 100

        source_row = source_metrics[source_metrics["category"] == category].iloc[0]
        deltas = {
            column: metrics[column] - float(source_row[column])
            for column in METRIC_COLUMNS
        }
        max_delta = max(abs(value) for value in deltas.values())
        assert max_delta <= 0.05, (
            f"{category} lệch E02 quá 0.05 điểm phần trăm: {max_delta}"
        )

        metrics["deltas_from_e02"] = deltas
        metrics["max_absolute_delta_from_e02"] = max_delta
        metrics["elapsed_seconds"] = elapsed_seconds
        metrics["heatmap_count"] = len(heatmaps)
        metrics["per_image_score_count"] = len(scores)
        metrics["attempt_root"] = str(attempt_root)

        shutil.copy2(score_file, attempt_root / "per-image-scores.csv")
        (attempt_root / "metrics.json").write_text(
            json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
        )
        (attempt_root / "SUCCESS.json").write_text(
            json.dumps(status, indent=2) + "\n", encoding="utf-8"
        )
    except Exception as error:
        status["validation_error"] = repr(error)
        (attempt_root / "FAILED.json").write_text(
            json.dumps(status, indent=2) + "\n", encoding="utf-8"
        )
        raise

    print(
        f"DONE {category}: {elapsed_seconds / 60:.2f} phút, "
        f"{metrics['heatmap_count']} heatmap"
    )
    print("Max delta from E02:", max_delta)
    return metrics
```

### Cell Q4 - Kiểm tra tiến độ

Chạy cell này trước và sau khi runtime bị ngắt.

```python
from collections import deque

progress_rows = []
for category in QUAL_CATEGORIES:
    category_dir = CATEGORY_ROOT / category
    attempts = sorted(category_dir.glob("attempt_*"))
    success = successful_qual_attempt(category)

    if success is not None:
        status_name = "SUCCESS"
        latest = success
    elif attempts:
        latest = attempts[-1]
        status_name = "FAILED" if (latest / "FAILED.json").is_file() else "INTERRUPTED"
    else:
        latest = None
        status_name = "NOT_STARTED"

    progress_rows.append(
        {
            "category": category,
            "status": status_name,
            "attempts": len(attempts),
            "latest_attempt": str(latest) if latest else "",
        }
    )

progress = pd.DataFrame(progress_rows)
display(progress)

for _, row in progress[progress["status"].isin(["FAILED", "INTERRUPTED"])].iterrows():
    log_path = Path(row["latest_attempt"]) / "console.log"
    if log_path.is_file():
        with log_path.open("r", encoding="utf-8", errors="replace") as stream:
            tail = list(deque(stream, maxlen=12))
        print(f"\n===== Cuối log của {row['category']} =====")
        print("".join(tail))
```

### Cell Q5 - Chạy hoặc resume bốn category

```python
qual_metrics = {}
for index, category in enumerate(QUAL_CATEGORIES, start=1):
    print(f"\n===== [{index}/{len(QUAL_CATEGORIES)}] {category} =====")
    qual_metrics[category] = run_qual_category(category)

print("Đã hoàn thành inference định tính cho bốn category VisA.")
```

Nếu một category thất bại, đọc `FAILED.json` và cuối `console.log`. Chỉ rerun sau khi đã hiểu nguyên nhân. Khi chạy lại Q5 theo quyết định của bạn, category có `SUCCESS.json` sẽ được bỏ qua và category chưa thành công sẽ tạo attempt mới.

### Cell Q6 - Ghép score, mask, heatmap và chọn 24 case

Cell này dùng `pixel_auroc_raw` đã tính từ raw anomaly map. Nó không đảo màu từ heatmap, nên không bị ảnh hưởng bởi việc VisA có thể lưu heatmap dưới dạng JPEG.

```python
import cv2
import numpy as np
from pathlib import Path

meta = json.loads((DATA_ROOT / "meta.json").read_text(encoding="utf-8"))

case_records = []
for category in QUAL_CATEGORIES:
    attempt_root = successful_qual_attempt(category)
    assert attempt_root is not None

    scores = pd.read_csv(attempt_root / "per-image-scores.csv")
    result_root = (
        attempt_root / "output" / "visa" / "ViT-L-14-336" /
        "imagesize518"
    )
    heatmap_root = result_root / category

    sample_by_image = {
        str(DATA_ROOT / sample["img_path"]): sample
        for sample in meta["test"][category]
    }
    assert len(sample_by_image) == len(scores)

    for _, score_row in scores.iterrows():
        image_path = Path(score_row["image_path"])
        image_key = str(image_path)
        assert image_key in sample_by_image, f"Không tìm thấy meta cho {image_path}"
        sample = sample_by_image[image_key]

        anomaly_type = image_path.parent.name
        heatmap_path = heatmap_root / anomaly_type / image_path.name
        assert heatmap_path.is_file(), f"Thiếu heatmap: {heatmap_path}"

        mask_path = ""
        if sample["mask_path"]:
            candidate_mask = DATA_ROOT / sample["mask_path"]
            assert candidate_mask.is_file(), f"Thiếu mask: {candidate_mask}"
            mask_path = str(candidate_mask)

        case_records.append(
            {
                **score_row.to_dict(),
                "anomaly_type": anomaly_type,
                "mask_path": mask_path,
                "heatmap_path": str(heatmap_path),
            }
        )

case_scores = pd.DataFrame(case_records)
case_scores.to_csv(RUN_ROOT / "qualitative-case-scores.csv", index=False)

def select_unique(rows, column, count, ascending, reason, used_paths):
    candidates = rows.sort_values(
        [column, "image_path"],
        ascending=[ascending, True],
        kind="mergesort",
    )
    candidates = candidates[~candidates["image_path"].isin(used_paths)]
    chosen = candidates.head(count).copy()
    assert len(chosen) == count, (
        f"Không đủ {count} case cho {reason}; chỉ còn {len(chosen)}"
    )
    chosen["selection_reason"] = reason
    used_paths.update(chosen["image_path"].tolist())
    return chosen

selected_parts = []
for category in QUAL_CATEGORIES:
    category_rows = case_scores[case_scores["category"] == category].copy()
    anomaly_rows = category_rows[category_rows["is_anomaly"].astype(int) == 1]
    normal_rows = category_rows[category_rows["is_anomaly"].astype(int) == 0]
    assert anomaly_rows["pixel_auroc_raw"].notna().all()

    used_paths = set()
    false_negative_candidates = select_unique(
        anomaly_rows,
        "rscin_score",
        2,
        True,
        "false_negative_candidate",
        used_paths,
    )
    false_positive_candidates = select_unique(
        normal_rows,
        "rscin_score",
        2,
        False,
        "false_positive_candidate",
        used_paths,
    )
    best_localization = select_unique(
        anomaly_rows,
        "pixel_auroc_raw",
        1,
        False,
        "best_localization",
        used_paths,
    )
    worst_localization = select_unique(
        anomaly_rows,
        "pixel_auroc_raw",
        1,
        True,
        "worst_localization",
        used_paths,
    )

    selected_parts.extend(
        [
            false_negative_candidates,
            false_positive_candidates,
            best_localization,
            worst_localization,
        ]
    )

selected = pd.concat(selected_parts, ignore_index=True)
assert len(selected) == 24
assert selected["image_path"].is_unique
assert set(selected["category"]) == set(QUAL_CATEGORIES)
assert (selected.groupby("category").size() == 6).all()

selected.to_csv(RUN_ROOT / "selected-qualitative-cases.csv", index=False)
display(
    selected[
        [
            "category",
            "anomaly_type",
            "image_path",
            "selection_reason",
            "initial_normalized_score",
            "rscin_score",
            "rscin_delta_from_initial_normalized",
            "pixel_auroc_raw",
        ]
    ]
)
```

`pixel_auroc_raw` là metric per-image dùng để xếp hạng case. Nó không thay thế Pixel AUROC chính thức của category, vốn được tính trên toàn bộ pixel của test split.

### Cell Q7 - Tạo asset và panel cho báo cáo

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
        [518, 518],
        interpolation=InterpolationMode.BICUBIC,
        antialias=True,
    )
    image = TF.center_crop(image, [518, 518])
    return np.asarray(image)

asset_records = []
for _, row in selected.iterrows():
    image_path = Path(row["image_path"])
    key = f"{row['category']}__{row['anomaly_type']}__{image_path.stem}"
    case_root = ASSET_ROOT / key
    case_root.mkdir(parents=True, exist_ok=True)

    original = prepare_original(image_path)
    heatmap_bgr = cv2.imread(row["heatmap_path"], cv2.IMREAD_COLOR)
    assert heatmap_bgr is not None, row["heatmap_path"]
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    if heatmap_rgb.shape[:2] != original.shape[:2]:
        heatmap_rgb = cv2.resize(
            heatmap_rgb,
            (original.shape[1], original.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )

    if row["mask_path"]:
        mask = cv2.imread(row["mask_path"], cv2.IMREAD_GRAYSCALE)
        assert mask is not None, row["mask_path"]
        mask = cv2.resize(
            mask,
            (original.shape[1], original.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
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

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.6))
    axes[0].imshow(original)
    axes[0].set_title("Original")
    # VisA masks store class IDs (1-7), so raw display on a 0-255 scale looks black.
    axes[1].imshow(mask > 0, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Ground truth")
    axes[2].imshow(heatmap_rgb)
    axes[2].set_title("MuSc heatmap")
    axes[3].imshow(overlay)
    axes[3].set_title("Overlay + GT contour")
    for axis in axes:
        axis.axis("off")

    pixel_auc = row["pixel_auroc_raw"]
    pixel_text = "N/A" if pd.isna(pixel_auc) else f"{pixel_auc:.4f}"
    fig.suptitle(
        f"{row['category']} | {row['anomaly_type']} | {image_path.name} | "
        f"{row['selection_reason']}\n"
        f"initial_norm={row['initial_normalized_score']:.4f} | "
        f"RsCIN={row['rscin_score']:.4f} | "
        f"delta={row['rscin_delta_from_initial_normalized']:+.4f} | "
        f"per-image pixel AUROC={pixel_text}",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.88])
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
assert len(asset_index) == 24
asset_index.to_csv(RUN_ROOT / "selected-assets-index.csv", index=False)
print("Đã tạo", len(asset_index), "panel tại", ASSET_ROOT)
```

### Cell Q8 - Xem nhanh các panel

```python
from IPython.display import display

for category in QUAL_CATEGORIES:
    print(f"\n===== {category} =====")
    rows = asset_index[asset_index["category"] == category]
    for _, row in rows.iterrows():
        print(
            row["selection_reason"],
            row["anomaly_type"],
            Path(row["image_path"]).name,
            f"RsCIN={row['rscin_score']:.4f}",
        )
        display(Image.open(row["panel_path"]))
```

Khi xem panel, ghi nhận riêng cho từng case:

- Heatmap có phủ đúng mask hay chỉ phản hồi trên biên hoặc cấu trúc lặp lại.
- Defect nhỏ, mảnh hoặc phân tán có bị bỏ sót hay không.
- Normal image có phản hồi mạnh trên linh kiện, đường mạch, background hoặc vùng phản sáng hay không.
- RsCIN làm score tăng hay giảm so với initial normalized score.
- `pcb2` và `pcb3` có pattern chung nào giải thích image-level degradation hay không.
- `pcb4` có separation tốt hơn giữa normal và anomaly hay không.
- `fryum` có localization hợp lý, phù hợp với Pixel AUROC khoảng 97.93 hay không.

### Cell Q9 - Kiểm tra cuối và tạo ZIP nhỏ

ZIP bỏ toàn bộ thư mục `output` chứa 752 heatmap trung gian. Bản sao `per-image-scores.csv` nằm ngoài `output`, nên vẫn được giữ trong ZIP.

```python
import zipfile

successful_attempts = {
    category: successful_qual_attempt(category)
    for category in QUAL_CATEGORIES
}
assert all(path is not None for path in successful_attempts.values())
assert len(asset_index) == 24
assert asset_index["panel_path"].map(lambda path: Path(path).is_file()).all()

metrics_records = [
    json.loads((attempt / "metrics.json").read_text(encoding="utf-8"))
    for attempt in successful_attempts.values()
]
max_metric_delta = max(
    record["max_absolute_delta_from_e02"] for record in metrics_records
)
assert max_metric_delta <= 0.05

summary_lines = [
    "# E02Q VisA Qualitative Summary",
    "",
    f"- Run ID: `{RUN_ID}`",
    f"- Source E02 Run ID: `{SOURCE_E02_RUN_ID}`",
    f"- Categories: {', '.join(QUAL_CATEGORIES)}",
    f"- Selected cases: {len(asset_index)}",
    f"- Maximum absolute metric delta from E02: {max_metric_delta:.12g} percentage points",
    "",
    "## Selection Counts",
    "",
]
for reason, count in asset_index["selection_reason"].value_counts().sort_index().items():
    summary_lines.append(f"- {reason}: {int(count)}")

(RUN_ROOT / "E02Q_SUMMARY.md").write_text(
    "\n".join(summary_lines) + "\n",
    encoding="utf-8",
)

done = {
    "run_id": RUN_ID,
    "source_e02_run_id": SOURCE_E02_RUN_ID,
    "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    "categories": QUAL_CATEGORIES,
    "selected_case_count": int(len(asset_index)),
    "max_absolute_metric_delta_from_e02": float(max_metric_delta),
    "status": "completed",
}
(RUN_ROOT / "E02Q_DONE.json").write_text(
    json.dumps(done, indent=2) + "\n", encoding="utf-8"
)

if (
    ACTIVE_Q_RUN_FILE.is_file()
    and ACTIVE_Q_RUN_FILE.read_text(encoding="utf-8").strip() == RUN_ID
):
    ACTIVE_Q_RUN_FILE.unlink()

zip_path = EXPORT_ROOT / f"{RUN_ID}.zip"
with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
    for path in RUN_ROOT.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(RUN_ROOT)
        if "output" in relative.parts:
            continue
        archive.write(path, arcname=str(relative))

with zipfile.ZipFile(zip_path) as archive:
    bad_member = archive.testzip()
    assert bad_member is None, f"ZIP lỗi tại: {bad_member}"
    names = archive.namelist()
    assert "E02Q_DONE.json" in names
    assert "selected-qualitative-cases.csv" in names
    assert "selected-assets-index.csv" in names
    assert not any("/output/" in f"/{name}" for name in names)

print("E02Q hoàn tất.")
print("ZIP:", zip_path)
print("Dung lượng ZIP:", zip_path.stat().st_size / 2**20, "MiB")
```

Tải ZIP trong `MyDrive/CS2203_MuSc/exports/` và gửi để kiểm tra. Không xóa thư mục E02Q trên Drive vì toàn bộ heatmap trung gian vẫn nằm ở đó.

## 6. Khôi phục khi runtime E02 đã mất

Nếu Cell Q1 báo runtime chưa đủ điều kiện, giữ nguyên E02Q Run ID vừa tạo và thực hiện:

1. Trong [hướng dẫn E02](2026-10-12-huong-dan-colab-musc-e02-visa.md), chạy Cell E02-2 để cài dependency và xác nhận GPU.
2. Chạy Cell E02-3 để khôi phục VisA từ Drive cache hoặc tải từ Amazon S3, xác minh TAR và SHA-256, rồi giải nén vào `/content`.
3. Chạy Cell E02-4 để clone đúng MuSc commit, sửa import và tạo `meta.json`.
4. Chạy Cell E02-4M để áp dụng CPU-memory patch.
5. Chạy Cell E02-5 để khôi phục CLIP weights.
6. Quay lại guide này, chạy Q1, Q2, Q3, Q4 và Q5.

Không chạy Cell E02-1 vì E02 đã aggregated xong. Cell Q0 của guide này đã tạo đúng `RUN_ROOT` và `SESSION_ROOT` cho E02Q.

Sau khi Colab bị ngắt giữa E02Q:

1. Chạy lại Q0 để resume đúng Run ID.
2. Thực hiện lại các bước khôi phục ở trên.
3. Chạy lại Q1 đến Q4.
4. Kiểm tra log của attempt bị ngắt.
5. Nếu quyết định tiếp tục, chạy Q5. Category đã có `SUCCESS.json` sẽ được bỏ qua.

## 7. Tiêu chí chấp nhận E02Q

E02Q đạt khi:

- [ ] Bốn category đều có `SUCCESS.json`.
- [ ] Có đúng 752 heatmap: 200 `pcb2`, 201 `pcb3`, 201 `pcb4` và 150 `fryum`.
- [ ] Mỗi category có đủ per-image score và không trùng `image_path`.
- [ ] Mỗi metric rerun lệch E02 không quá 0.05 điểm phần trăm.
- [ ] Có đúng 24 selected case, sáu case khác nhau cho mỗi category.
- [ ] Mỗi category có hai `false_negative_candidate`, hai `false_positive_candidate`, một `best_localization` và một `worst_localization`.
- [ ] Mỗi selected case có original, mask, heatmap, overlay và panel.
- [ ] Candidate được mô tả là candidate, không bị trình bày như lỗi phân loại đã xác nhận.
- [ ] `fryum` được dùng để đánh giá tính hợp lý của localization, không được dùng để tuyên bố giá trị official chính xác là 97.89.
- [ ] ZIP nhỏ không chứa thư mục `output`, vượt qua `testzip()` và được tải về để kiểm tra.

## 8. Artifact cần gửi sau khi chạy

Gửi file:

```text
/Users/duykhanh/Downloads/<E02Q_RUN_ID>.zip
```

Không cần gửi toàn bộ thư mục heatmap trên Drive. ZIP nhỏ đã chứa đủ bằng chứng để kiểm tra selection protocol, metric continuity, instrumentation và 24 panel.
