"""Shared immutable data contracts for the MuSc classroom demo."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


Severity = Literal["info", "warning", "error"]
BatchStatus = Literal["blocked", "experimental", "recommended"]


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    severity: Severity
    message_vi: str
    filenames: tuple[str, ...] = ()


@dataclass(frozen=True)
class ValidatedImage:
    image_id: str
    source_name: str
    path: Path
    sha256: str


@dataclass(frozen=True)
class BatchValidationReport:
    images: tuple[ValidatedImage, ...]
    issues: tuple[ValidationIssue, ...]
    status: BatchStatus
    can_run: bool
