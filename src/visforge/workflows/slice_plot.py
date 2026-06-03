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
    show_mesh: bool = False,
    mesh_color: str = "white",
    mesh_linewidth: float = 0.15,
    mesh_alpha: float = 0.75,
    mesh_max_lines: int | None = None,
) -> PlotResult:
    dataset = open_dataset(path, backend=backend)
    slice_data = dataset.read_slice(field, iteration=iteration, plane=plane)
    return plot_scalar_slice(
        slice_data,
        output=output,
        show_mesh=show_mesh,
        mesh_color=mesh_color,
        mesh_linewidth=mesh_linewidth,
        mesh_alpha=mesh_alpha,
        mesh_max_lines=mesh_max_lines,
    )
