"""Pure Vietnamese presentation helpers for the MuSc demo UI."""

from __future__ import annotations

from dataclasses import dataclass

from .inference import InferenceResult, InferenceStageError
from .models import BatchValidationReport
from .prepared_results import PreparedCatalog, PreparedRun


SCOPE_CARD_MARKDOWN = """
## Phạm vi bài toán

MuSc thực hiện **phát hiện bất thường và định vị vùng nghi bất thường** theo cách
training-free, transductive. Kết quả của một ảnh phụ thuộc vào các ảnh còn lại
trong cùng tập.

**Input**

- Một **tập 5-200 ảnh JPEG/PNG chưa có nhãn**, không phải một ảnh đơn lẻ.
- Tất cả ảnh phải thuộc **một category sản phẩm công nghiệp** và có điều kiện
  chụp tương đối nhất quán.
- 5-39 ảnh chỉ dành cho thử nghiệm; 40-200 ảnh là khoảng khuyến nghị.

**Output**

- anomaly score tương đối cho từng ảnh và danh sách xếp hạng mức độ đáng ngờ.
- heatmap và overlay của vùng nghi bất thường.

**Không hỗ trợ / không được suy diễn**

- Không xử lý một ảnh đơn lẻ hoặc trộn nhiều category trong cùng tập.
- Không tự kiểm tra category hay bối cảnh công nghiệp của ảnh.
- Không tạo nhãn kết luận, ngưỡng triển khai hay benchmark metric cho tập tải lên.
- Không xác nhận khả năng dùng cho ảnh y khoa hoặc nha khoa và không đưa ra chẩn đoán.
""".strip()


@dataclass(frozen=True)
class EnvironmentCardState:
    markdown: str
    live_available: bool
    prepared_available: bool
    initialize_interactive: bool


@dataclass(frozen=True)
class ControlState:
    interactive: bool
    reason: str


@dataclass(frozen=True)
class WarningState:
    markdown: str
    visible: bool
    variant: str


def environment_card(
    *,
    gpu_available: bool,
    gpu_name: str | None,
    engine_state: str,
    prepared: PreparedCatalog,
) -> EnvironmentCardState:
    if gpu_available:
        gpu_status = f"Sẵn sàng: `{gpu_name or 'CUDA GPU'}`"
    else:
        gpu_status = "Không có CUDA GPU - live inference bị khóa"

    model_labels = {
        "uninitialized": "Chưa khởi tạo",
        "ready": "Sẵn sàng",
        "failed": "Khởi tạo thất bại",
    }
    prepared_available = bool(prepared.datasets())
    if prepared.error:
        prepared_status = "Không khả dụng"
    elif prepared_available:
        prepared_status = "Sẵn sàng"
    else:
        prepared_status = "Chưa gắn artifact"

    markdown = "\n".join(
        (
            "## Trạng thái môi trường",
            f"- **GPU:** {gpu_status}",
            f"- **MuSc model:** {model_labels.get(engine_state, engine_state)}",
            f"- **Prepared Benchmark:** {prepared_status}",
            "- Live inference trên Tesla T4 có thể mất vài phút.",
        )
    )
    return EnvironmentCardState(
        markdown=markdown,
        live_available=gpu_available and engine_state == "ready",
        prepared_available=prepared_available,
        initialize_interactive=gpu_available and engine_state != "ready",
    )


def run_button_state(
    report: BatchValidationReport | None,
    *,
    confirmed: bool,
    engine_state: str,
) -> ControlState:
    if engine_state != "ready":
        return ControlState(False, "MuSc model chưa sẵn sàng.")
    if report is None:
        return ControlState(False, "Hãy tải lên và kiểm tra một tập ảnh.")
    if not report.can_run:
        return ControlState(False, "Tập ảnh chưa đạt điều kiện kỹ thuật.")
    if not confirmed:
        return ControlState(False, "Cần xác nhận giả định về dữ liệu đầu vào.")
    return ControlState(True, "Có thể chạy live inference.")


def validation_markdown(report: BatchValidationReport | None) -> str:
    if report is None:
        return "Chưa có tập ảnh được kiểm tra."
    status_labels = {
        "blocked": "Bị khóa",
        "experimental": "Thử nghiệm",
        "recommended": "Khuyến nghị",
    }
    lines = [
        "### Kết quả kiểm tra tập ảnh",
        f"- **Ảnh hợp lệ:** {len(report.images)}",
        f"- **Trạng thái:** {status_labels[report.status]}",
    ]
    for issue in report.issues:
        files = f" ({', '.join(issue.filenames)})" if issue.filenames else ""
        lines.append(f"- `{issue.code}`: {issue.message_vi}{files}")
    return "\n".join(lines)


def experimental_warning(report: BatchValidationReport | None) -> WarningState:
    if report is None or report.status != "experimental":
        return WarningState("", False, "danger")
    return WarningState(
        "**Cảnh báo:** Tập 5-39 ảnh chỉ dành cho thử nghiệm. "
        "So sánh transductive có thể thiếu ổn định; không dùng kết quả làm kết luận.",
        True,
        "danger",
    )


def custom_results_markdown(result: InferenceResult) -> str:
    warnings = ""
    if result.warnings:
        warnings = "\n\nCảnh báo: " + ", ".join(result.warnings)
    return (
        "### Kết quả live inference\n\n"
        f"Đã xếp hạng **{len(result.items)} ảnh** bằng anomaly score tương đối. "
        "Giá trị chỉ có ý nghĩa khi so sánh trong chính tập ảnh này. "
        "Mỗi ảnh có ảnh gốc, heatmap và overlay vùng nghi bất thường."
        f"{warnings}"
    )


def prepared_result_markdown(run: PreparedRun) -> str:
    metrics = "\n".join(
        f"- **{name}:** {value:.4f}" for name, value in run.metrics.items()
    )
    if not metrics:
        metrics = "- Không có metric trong artifact này."
    notices = ""
    if run.notices:
        notices = "\n\n" + "\n".join(f"- {notice}" for notice in run.notices)
    return (
        f"### {run.badge}\n\n"
        f"- **Run ID:** `{run.run_id}`\n"
        f"- **Experiment:** `{run.experiment_id}`\n"
        f"- **Dataset / category:** {run.dataset} / {run.category}\n"
        f"- **Mô tả:** {run.description}\n\n"
        f"**Metric đã ghi nhận trong cùng run**\n\n{metrics}{notices}"
    )


def pipeline_error_markdown(error: BaseException) -> str:
    stage_names = {
        "normalize": "chuẩn hóa ảnh",
        "clip": "CLIP feature extraction",
        "lnamd": "LNAMD",
        "msm": "MSM",
        "rscin": "RsCIN",
        "render": "tạo heatmap và overlay",
    }
    if isinstance(error, InferenceStageError):
        stage = stage_names.get(error.stage, error.stage)
        if error.code == "CUDA_OOM":
            recovery = "Hãy giảm số lượng ảnh rồi thử lại."
        else:
            recovery = "Hãy kiểm tra tập ảnh, trạng thái model và thử lại."
        return f"**Không thể hoàn tất bước {stage}.** {recovery}"
    return "**Live inference không hoàn tất.** Hãy kiểm tra môi trường và thử lại."
