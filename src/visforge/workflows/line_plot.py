"""High-level line plot workflow."""

from __future__ import annotations

from pathlib import Path

from visforge.data.registry import open_dataset
from visforge.plotting.base import PlotLabels, PlotResult
from visforge.plotting.line import plot_line


def make_line_plot(
    path: str | Path,
    *,
    field: str,
    axis: str,
    iteration: int | None = None,
    backend: str = "tsv",
    output: str | Path | None = None,
    labels: PlotLabels | None = None,
) -> PlotResult:
    dataset = open_dataset(path, backend=backend)
    line = dataset.read_line(field, iteration=iteration, axis=axis)
    return plot_line(line, output=output, labels=labels)
