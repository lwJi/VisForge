"""High-level scalar slice plot workflow."""

from __future__ import annotations

from pathlib import Path

from visforge.data.registry import open_dataset
from visforge.data.model import PlaneSpec
from visforge.data.plane import sample_field_on_plane
from visforge.plotting.base import PlotResult
from visforge.plotting.scalar import plot_scalar_slice


def make_slice_plot(
    path: str | Path,
    *,
    field: str,
    component: str | None = None,
    iteration: int | None = None,
    plane: str | None = None,
    sample_plane: PlaneSpec | None = None,
    backend: str = "auto",
    output: str | Path | None = None,
    show_mesh: bool = False,
    mesh_color: str = "white",
    mesh_linewidth: float = 0.15,
    mesh_alpha: float = 0.75,
    mesh_max_lines: int | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    scale: str = "linear",
    vmin: float | None = None,
    vmax: float | None = None,
) -> PlotResult:
    if plane is not None and sample_plane is not None:
        raise ValueError("Use either an axis-aligned plane or sample_plane, not both.")
    dataset = open_dataset(path, backend=backend)
    if sample_plane is None:
        slice_data = dataset.read_slice(field, component=component, iteration=iteration, plane=plane)
    else:
        field_data = dataset.read_field(field, component=component, iteration=iteration)
        slice_data = sample_field_on_plane(field_data, sample_plane)
    return plot_scalar_slice(
        slice_data,
        output=output,
        show_mesh=show_mesh,
        mesh_color=mesh_color,
        mesh_linewidth=mesh_linewidth,
        mesh_alpha=mesh_alpha,
        mesh_max_lines=mesh_max_lines,
        xlim=xlim,
        ylim=ylim,
        scale=scale,
        vmin=vmin,
        vmax=vmax,
    )
