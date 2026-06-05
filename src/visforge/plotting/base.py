"""Shared plotting return types."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PlotLabels:
    """User-provided plot text overrides."""

    title: str | None = None
    xlabel: str | None = None
    ylabel: str | None = None
    legend: str | None = None
    colorbar: str | None = None


@dataclass(frozen=True)
class PlotResult:
    """Result of creating a plot."""

    figure: Any
    axes: Any
    output: Path | None = None
