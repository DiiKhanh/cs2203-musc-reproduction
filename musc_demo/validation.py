"""Secure upload expansion and validation for unlabeled image batches."""

from __future__ import annotations

import hashlib
import shutil
import stat
import zipfile
from collections.abc import Sequence
from pathlib import Path, PurePosixPath

from PIL import Image, UnidentifiedImageError

from .models import BatchStatus, BatchValidationReport, ValidatedImage, ValidationIssue


SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
FATAL_ISSUE_CODES = {"UNSAFE_ZIP_ENTRY", "CORRUPT_ARCHIVE"}


def _issue(
    code: str,
    severity: str,
    message_vi: str,
    filenames: Sequence[str] = (),
) -> ValidationIssue:
    return ValidationIssue(
        code=code,
        severity=severity,  # type: ignore[arg-type]
        message_vi=message_vi,
        filenames=tuple(filenames),
    )


def _is_zip_symlink(member: zipfile.ZipInfo) -> bool:
    mode = member.external_attr >> 16
    return stat.S_ISLNK(mode)


def _safe_member_path(name: str) -> PurePosixPath | None:
    normalized = name.replace("\\", "/")
    path = PurePosixPath(normalized)
    if path.is_absolute() or ".." in path.parts or not path.name:
        return None
    return path


def _extract_archive(
    archive: Path, destination: Path
) -> tuple[list[tuple[Path, str]], list[ValidationIssue]]:
    extracted: list[tuple[Path, str]] = []
    issues: list[ValidationIssue] = []
    try:
        with zipfile.ZipFile(archive) as zip_file:
            members = [member for member in zip_file.infolist() if not member.is_dir()]
            unsafe = [
                member.filename
                for member in members
                if _safe_member_path(member.filename) is None or _is_zip_symlink(member)
            ]
            if unsafe:
                return [], [
                    _issue(
                        "UNSAFE_ZIP_ENTRY",
                        "error",
                        "Tệp ZIP chứa đường dẫn không an toàn hoặc symbolic link.",
                        unsafe,
                    )
                ]

            for member in members:
                relative = _safe_member_path(member.filename)
                assert relative is not None
                target = destination.joinpath(*relative.parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                with zip_file.open(member) as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output)
                extracted.append((target, relative.as_posix()))
    except (OSError, zipfile.BadZipFile):
        issues.append(
            _issue(
                "CORRUPT_ARCHIVE",
                "error",
                "Không thể đọc tệp ZIP.",
                (archive.name,),
            )
        )
    return extracted, issues


def _collect_candidates(
    inputs: Sequence[Path], workspace: Path
) -> tuple[list[tuple[Path, str]], list[ValidationIssue]]:
    candidates: list[tuple[Path, str]] = []
    issues: list[ValidationIssue] = []
    incoming = workspace / "incoming"
    incoming.mkdir(parents=True, exist_ok=True)

    for index, raw_path in enumerate(inputs):
        path = Path(raw_path)
        if path.suffix.lower() == ".zip":
            archive_candidates, archive_issues = _extract_archive(
                path, incoming / f"archive-{index:03d}"
            )
            candidates.extend(archive_candidates)
            issues.extend(archive_issues)
            continue

        if path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
            issues.append(
                _issue(
                    "UNSUPPORTED_FILE",
                    "warning",
                    "Tệp không thuộc định dạng JPEG hoặc PNG được hỗ trợ.",
                    (path.name,),
                )
            )
            continue

        destination = incoming / f"file-{index:03d}" / path.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copyfile(path, destination)
        except OSError:
            issues.append(
                _issue(
                    "CORRUPT_IMAGE",
                    "warning",
                    "Không thể đọc tệp ảnh.",
                    (path.name,),
                )
            )
            continue
        candidates.append((destination, path.name))

    return candidates, issues


def _validate_candidates(
    candidates: Sequence[tuple[Path, str]],
) -> tuple[list[ValidatedImage], list[ValidationIssue]]:
    accepted: list[ValidatedImage] = []
    issues: list[ValidationIssue] = []
    seen_hashes: dict[str, str] = {}
    prefix_counts: dict[str, int] = {}

    for path, source_name in candidates:
        if path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
            issues.append(
                _issue(
                    "UNSUPPORTED_FILE",
                    "warning",
                    "Tệp không thuộc định dạng JPEG hoặc PNG được hỗ trợ.",
                    (source_name,),
                )
            )
            continue

        try:
            with Image.open(path) as image:
                image.verify()
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
        except (OSError, UnidentifiedImageError):
            issues.append(
                _issue(
                    "CORRUPT_IMAGE",
                    "warning",
                    "Không thể giải mã tệp ảnh.",
                    (source_name,),
                )
            )
            continue

        if digest in seen_hashes:
            issues.append(
                _issue(
                    "DUPLICATE_IMAGE",
                    "warning",
                    "Ảnh trùng hoàn toàn đã được loại khỏi tập chạy.",
                    (source_name,),
                )
            )
            continue

        prefix = digest[:12]
        ordinal = prefix_counts.get(prefix, 0) + 1
        prefix_counts[prefix] = ordinal
        seen_hashes[digest] = source_name
        accepted.append(
            ValidatedImage(
                image_id=f"{prefix}-{ordinal:03d}",
                source_name=source_name,
                path=path,
                sha256=digest,
            )
        )

    return accepted, issues


def _batch_status(count: int) -> tuple[BatchStatus, bool, ValidationIssue | None]:
    if count < 5:
        return (
            "blocked",
            False,
            _issue(
                "BATCH_TOO_SMALL",
                "error",
                "MuSc cần ít nhất 5 ảnh hợp lệ trong cùng một category.",
            ),
        )
    if count < 40:
        return (
            "experimental",
            True,
            _issue(
                "BATCH_EXPERIMENTAL",
                "warning",
                "Tập 5-39 ảnh chỉ phù hợp để thử nghiệm; kết quả có thể thiếu ổn định.",
            ),
        )
    if count <= 200:
        return "recommended", True, None
    return (
        "blocked",
        False,
        _issue(
            "BATCH_TOO_LARGE",
            "error",
            "Tập ảnh vượt quá giới hạn 200 ảnh của demo.",
        ),
    )


def validate_uploads(
    inputs: Sequence[Path], workspace: Path
) -> BatchValidationReport:
    """Expand and validate one upload selection inside an isolated workspace."""

    workspace = Path(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    candidates, collection_issues = _collect_candidates(inputs, workspace)
    images, image_issues = _validate_candidates(candidates)
    status, count_allows_run, batch_issue = _batch_status(len(images))
    issues = [*collection_issues, *image_issues]
    if batch_issue is not None:
        issues.append(batch_issue)
    has_fatal_issue = any(issue.code in FATAL_ISSUE_CODES for issue in issues)
    if has_fatal_issue:
        status = "blocked"

    return BatchValidationReport(
        images=tuple(images),
        issues=tuple(issues),
        status=status,
        can_run=count_allows_run and not has_fatal_issue,
    )


def confirmation_allows_run(
    report: BatchValidationReport, confirmed: bool
) -> bool:
    """Return whether the technical report and user attestation allow inference."""

    return report.can_run and confirmed

