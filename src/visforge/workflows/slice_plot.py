"""High-level scalar slice plot workflow."""

from __future__ import annotations

from pathlib import Path

from visforge.data.registry import open_dataset
from visforge.plotting.base import PlotResult
from visforge.plotting.scalar import plot_scalar_slice


def make_slice_plot(
    path: str | Path,
    *,
    field: str,
    iteration: int | None = None,
    plane: str | None = None,
    backend: str = "auto",
    output: str | Path | None = None,
) -> PlotResult:
    dataset = open_dataset(path, backend=backend)
    slice_data = dataset.read_slice(field, iteration=iteration, plane=plane)
    return plot_scalar_slice(slice_data, output=output)
