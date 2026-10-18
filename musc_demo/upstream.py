"""Version-checked imports from the official MuSc checkout."""

from __future__ import annotations

import importlib
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Callable


PINNED_MUSC_COMMIT = "b76b93da8bd3096a99964a96ae29d46f197a0651"


class UpstreamVersionError(RuntimeError):
    def __init__(self, expected: str, actual: str) -> None:
        self.expected = expected
        self.actual = actual
        super().__init__(
            f"MuSc checkout must be {expected}, but the current commit is {actual}"
        )


@dataclass(frozen=True)
class UpstreamModules:
    open_clip: ModuleType
    lnamd_factory: type
    msm: Callable[..., object]
    rscin: Callable[..., object]


def load_upstream_modules(musc_root: Path) -> UpstreamModules:
    """Verify and import the algorithm modules from one pinned MuSc checkout."""

    musc_root = Path(musc_root).resolve()
    result = subprocess.run(
        ["git", "-C", str(musc_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    actual_commit = result.stdout.strip()
    if actual_commit != PINNED_MUSC_COMMIT:
        raise UpstreamVersionError(PINNED_MUSC_COMMIT, actual_commit)

    root_text = str(musc_root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)

    open_clip = importlib.import_module("models.backbone.open_clip")
    lnamd_module = importlib.import_module("models.modules._LNAMD")
    msm_module = importlib.import_module("models.modules._MSM")
    rscin_module = importlib.import_module("models.modules._RsCIN")
    return UpstreamModules(
        open_clip=open_clip,
        lnamd_factory=lnamd_module.LNAMD,
        msm=msm_module.MSM,
        rscin=rscin_module.RsCIN,
    )

