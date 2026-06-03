"""Path and filename detection helpers."""

from __future__ import annotations

import re
from pathlib import Path

from visforge.data.model import BackendName

ITERATION_RE = re.compile(r"\.it(?P<iteration>\d+)")
PLANE_RE = re.compile(r"\.(?P<plane>xy|xz|yz)\.it\d+")
TSV_RE = re.compile(
    r"^(?P<prefix>.+)-(?P<variable>[^.]+)\.it(?P<iteration>\d+)\.(?P<axis>[xyz])\.tsv$"
)


def detect_backend(path: Path) -> BackendName | None:
    """Infer a backend from a file or directory path."""

    if path.suffix == ".bp5" or path.name.endswith(".bp5"):
        return "openpmd"
    if path.suffix == ".silo":
        return "silo"
    if path.suffix == ".tsv":
        return "tsv"
    return None


def parse_iteration(path: Path) -> int | None:
    """Parse CarpetX iteration numbers from names such as ``*.it00000120.bp5``."""

    match = ITERATION_RE.search(path.name)
    if match is None:
        return None
    return int(match.group("iteration"))


def parse_plane(path: Path) -> str | None:
    """Parse native 2D plane names such as ``xy`` from CarpetX data names."""

    match = PLANE_RE.search(path.name)
    if match is None:
        return None
    return match.group("plane")


def parse_tsv_name(path: Path) -> tuple[str, int, str] | None:
    """Return ``(variable, iteration, axis)`` for a CarpetX TSV line output."""

    match = TSV_RE.match(path.name)
    if match is None:
        return None
    return (
        match.group("variable"),
        int(match.group("iteration")),
        match.group("axis"),
    )


def is_silo_fragment(path: Path) -> bool:
    """Return true for per-rank payload files inside ``*.silo.dir`` directories."""

    return path.suffix == ".silo" and path.parent.name.endswith(".silo.dir")
