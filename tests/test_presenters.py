from pathlib import Path

from musc_demo.inference import InferenceItem, InferenceResult, InferenceStageError
from musc_demo.models import BatchValidationReport, ValidationIssue, ValidatedImage
from musc_demo.prepared_results import (
    PREPARED_BADGE,
    PreparedAssets,
    PreparedCatalog,
    PreparedExample,
    PreparedRun,
)
from musc_demo.presenters import (
    SCOPE_CARD_MARKDOWN,
    custom_results_markdown,
    environment_card,
    experimental_warning,
    pipeline_error_markdown,
    prepared_result_markdown,
    run_button_state,
)


def _report(*, status: str = "recommended", can_run: bool = True):
    image = ValidatedImage("id-1", "sample.png", Path("sample.png"), "abc")
    issues = ()
    if status == "experimental":
        issues = (
            ValidationIssue(
                "BATCH_EXPERIMENTAL",
                "warning",
                "Tập thử nghiệm có thể thiếu ổn định.",
            ),
        )
    return BatchValidationReport(
        images=(image,),
        issues=issues,
        status=status,  # type: ignore[arg-type]
        can_run=can_run,
    )


def _prepared_catalog(tmp_path: Path) -> PreparedCatalog:
    example = PreparedExample(
        run_id="run-1",
        image_id="image-1",
        source_name="sample.png",
        defect_type="scratch",
        role="success",
        raw_score=0.4,
        adjusted_score=0.6,
        assets=PreparedAssets(None, None, None, None),
        notices=(),
    )
    run = PreparedRun(
        run_id="run-1",
        experiment_id="E01",
        dataset="MVTec AD",
        category="bottle",
        timestamp="2026-10-18T00:00:00Z",
        source_commit="abc123",
        description="Prepared fixture",
        sample_counts={"normal": 2, "anomalous": 1},
        defect_types=("scratch",),
        ground_truth_available=True,
        metrics={"Image AUROC": 0.95},
        examples=(example,),
        available=True,
        notices=(),
    )
    return PreparedCatalog((run,))


def test_scope_card_states_input_output_and_non_goals():
    assert "Input" in SCOPE_CARD_MARKDOWN
    assert "một category" in SCOPE_CARD_MARKDOWN
    assert "Output" in SCOPE_CARD_MARKDOWN
    assert "anomaly score tương đối" in SCOPE_CARD_MARKDOWN
    assert "Không hỗ trợ" in SCOPE_CARD_MARKDOWN
    assert "một ảnh đơn lẻ" in SCOPE_CARD_MARKDOWN
    assert "y khoa" in SCOPE_CARD_MARKDOWN
    assert "transductive" in SCOPE_CARD_MARKDOWN


def test_environment_card_reports_gpu_model_and_prepared_status(tmp_path):
    state = environment_card(
        gpu_available=True,
        gpu_name="Tesla T4",
        engine_state="ready",
        prepared=_prepared_catalog(tmp_path),
    )

    assert "Tesla T4" in state.markdown
    assert "sẵn sàng" in state.markdown.lower()
    assert "Prepared Benchmark" in state.markdown
    assert state.live_available is True
    assert state.prepared_available is True


def test_run_button_requires_valid_batch_confirmation_and_ready_model():
    ready = run_button_state(_report(), confirmed=True, engine_state="ready")
    assert ready.interactive is True

    assert not run_button_state(
        _report(can_run=False), confirmed=True, engine_state="ready"
    ).interactive
    assert not run_button_state(
        _report(), confirmed=False, engine_state="ready"
    ).interactive
    assert not run_button_state(
        _report(), confirmed=True, engine_state="uninitialized"
    ).interactive


def test_experimental_batch_shows_red_warning():
    warning = experimental_warning(_report(status="experimental"))

    assert warning.visible is True
    assert warning.variant == "danger"
    assert "5-39" in warning.markdown
    assert "thiếu ổn định" in warning.markdown


def test_custom_results_do_not_contain_labels_metrics_or_ground_truth():
    result = InferenceResult(
        items=(
            InferenceItem(
                image_id="image-1",
                source_name="sample.png",
                path=Path("sample.png"),
                raw_score=0.4,
                adjusted_score=0.6,
                anomaly_map=__import__("numpy").zeros((2, 2)),
            ),
        ),
        warnings=(),
        config={"rscin_neighbors": [1, 2, 3]},
        upstream_commit="abc123",
    )

    content = custom_results_markdown(result).lower()
    assert "anomaly score tương đối" in content
    assert "normal" not in content
    assert "defect" not in content
    assert "ground truth" not in content
    assert "auroc" not in content
    assert "accuracy" not in content


def test_prepared_result_has_explicit_not_live_badge(tmp_path):
    run = _prepared_catalog(tmp_path).get_run("run-1")
    content = prepared_result_markdown(run)

    assert PREPARED_BADGE in content
    assert "Image AUROC" in content
    assert "run-1" in content


def test_gpu_failure_keeps_prepared_tab_available(tmp_path):
    state = environment_card(
        gpu_available=False,
        gpu_name=None,
        engine_state="failed",
        prepared=_prepared_catalog(tmp_path),
    )

    assert state.live_available is False
    assert state.prepared_available is True
    assert "không có cuda gpu" in state.markdown.lower()


def test_pipeline_error_hides_raw_traceback_and_names_failed_stage():
    error = InferenceStageError(
        "lnamd",
        "INFERENCE_FAILED",
        "Traceback /private/secret/model.py token=super-secret",
    )

    content = pipeline_error_markdown(error)
    assert "LNAMD" in content
    assert "thử lại" in content.lower()
    assert "Traceback" not in content
    assert "/private/secret" not in content
    assert "super-secret" not in content
