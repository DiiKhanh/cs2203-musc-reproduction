from __future__ import annotations

import io
import shutil
import stat
import zipfile
from pathlib import Path

import pytest
from PIL import Image

from musc_demo.validation import confirmation_allows_run, validate_uploads


def write_image(path: Path, color: tuple[int, int, int] = (20, 40, 60)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 6), color).save(path)
    return path


def png_bytes(color: tuple[int, int, int]) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (8, 6), color).save(buffer, format="PNG")
    return buffer.getvalue()


def issue_codes(report: object) -> set[str]:
    return {issue.code for issue in report.issues}


def test_rejects_one_image(tmp_path: Path) -> None:
    report = validate_uploads(
        [write_image(tmp_path / "one.png")], tmp_path / "workspace"
    )

    assert len(report.images) == 1
    assert report.status == "blocked"
    assert report.can_run is False
    assert "BATCH_TOO_SMALL" in issue_codes(report)


def test_warns_for_five_images(tmp_path: Path) -> None:
    inputs = [
        write_image(tmp_path / f"image-{index}.png", (index, 10, 20))
        for index in range(5)
    ]

    report = validate_uploads(inputs, tmp_path / "workspace")

    assert len(report.images) == 5
    assert report.status == "experimental"
    assert report.can_run is True
    warning = next(issue for issue in report.issues if issue.code == "BATCH_EXPERIMENTAL")
    assert warning.severity == "warning"


def test_accepts_forty_images(tmp_path: Path) -> None:
    inputs = [
        write_image(tmp_path / f"image-{index}.png", (index, index, index))
        for index in range(40)
    ]

    report = validate_uploads(inputs, tmp_path / "workspace")

    assert len(report.images) == 40
    assert report.status == "recommended"
    assert report.can_run is True
    assert "BATCH_EXPERIMENTAL" not in issue_codes(report)


def test_rejects_more_than_two_hundred_images(tmp_path: Path) -> None:
    inputs = [
        write_image(
            tmp_path / f"image-{index}.png",
            (index % 251, (index // 251) % 251, index % 239),
        )
        for index in range(201)
    ]

    report = validate_uploads(inputs, tmp_path / "workspace")

    assert len(report.images) == 201
    assert report.status == "blocked"
    assert report.can_run is False
    assert "BATCH_TOO_LARGE" in issue_codes(report)


def test_reports_corrupt_unsupported_and_exact_duplicate_files(
    tmp_path: Path,
) -> None:
    original = write_image(tmp_path / "original.png")
    duplicate = tmp_path / "duplicate.png"
    shutil.copyfile(original, duplicate)
    corrupt = tmp_path / "corrupt.png"
    corrupt.write_bytes(b"not an image")
    unsupported = tmp_path / "notes.txt"
    unsupported.write_text("not an image", encoding="utf-8")

    report = validate_uploads(
        [original, duplicate, corrupt, unsupported], tmp_path / "workspace"
    )

    assert [image.source_name for image in report.images] == ["original.png"]
    assert issue_codes(report) >= {
        "CORRUPT_IMAGE",
        "UNSUPPORTED_FILE",
        "DUPLICATE_IMAGE",
        "BATCH_TOO_SMALL",
    }
    duplicate_issue = next(
        issue for issue in report.issues if issue.code == "DUPLICATE_IMAGE"
    )
    assert duplicate_issue.filenames == ("duplicate.png",)


@pytest.mark.parametrize("unsafe_kind", ["traversal", "symlink"])
def test_rejects_zip_path_traversal_and_symlink_entries(
    tmp_path: Path, unsafe_kind: str
) -> None:
    archive = tmp_path / f"{unsafe_kind}.zip"
    with zipfile.ZipFile(archive, "w") as zip_file:
        zip_file.writestr("safe.png", png_bytes((1, 2, 3)))
        if unsafe_kind == "traversal":
            zip_file.writestr("../escape.png", png_bytes((4, 5, 6)))
        else:
            member = zipfile.ZipInfo("linked.png")
            member.create_system = 3
            member.external_attr = (stat.S_IFLNK | 0o777) << 16
            zip_file.writestr(member, "safe.png")

    workspace = tmp_path / "workspace"
    report = validate_uploads([archive], workspace)

    assert report.can_run is False
    assert report.images == ()
    assert "UNSAFE_ZIP_ENTRY" in issue_codes(report)
    assert not (tmp_path / "escape.png").exists()


def test_assigns_unique_ids_to_duplicate_basenames(tmp_path: Path) -> None:
    archive = tmp_path / "batch.zip"
    with zipfile.ZipFile(archive, "w") as zip_file:
        zip_file.writestr("line-a/image.png", png_bytes((1, 2, 3)))
        zip_file.writestr("line-b/image.png", png_bytes((4, 5, 6)))
        for index in range(3):
            zip_file.writestr(
                f"line-c/image-{index}.png", png_bytes((10 + index, 20, 30))
            )

    first = validate_uploads([archive], tmp_path / "workspace-1")
    second = validate_uploads([archive], tmp_path / "workspace-2")

    assert len(first.images) == 5
    assert len({image.image_id for image in first.images}) == 5
    assert [image.source_name for image in first.images[:2]] == [
        "line-a/image.png",
        "line-b/image.png",
    ]
    assert [image.image_id for image in first.images] == [
        image.image_id for image in second.images
    ]


def test_requires_user_confirmation_after_valid_batch(tmp_path: Path) -> None:
    inputs = [
        write_image(tmp_path / f"image-{index}.png", (index, 40, 50))
        for index in range(5)
    ]
    valid_report = validate_uploads(inputs, tmp_path / "valid-workspace")
    blocked_report = validate_uploads(
        [write_image(tmp_path / "only-one.png", (99, 40, 50))],
        tmp_path / "blocked-workspace",
    )

    assert confirmation_allows_run(valid_report, confirmed=False) is False
    assert confirmation_allows_run(valid_report, confirmed=True) is True
    assert confirmation_allows_run(blocked_report, confirmed=True) is False
