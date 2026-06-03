"""Scalar 2D plotting for backend-neutral ``SliceData`` objects."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from visforge.data.model import GridBlock, SliceData
from visforge.plotting.base import PlotResult
from visforge.plotting.output import save_figure
from visforge.plotting.style import (
    DEFAULT_CMAP,
    DEFAULT_FIGSIZE,
    configure_matplotlib_environment,
    field_label,
)

DEFAULT_EXTREME_VALUE_LIMIT = 1.0e100


def plot_scalar_slice(
    data: SliceData,
    *,
    output: str | Path | None = None,
    title: str | None = None,
    cmap: str = DEFAULT_CMAP,
    vmin: float | None = None,
    vmax: float | None = None,
    show_mesh: bool = False,
    mesh_color: str = "white",
    mesh_linewidth: float = 0.8,
) -> PlotResult:
    """Plot one or more blocks from a scalar 2D slice."""

    configure_matplotlib_environment()
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    figure, axes = plt.subplots(figsize=DEFAULT_FIGSIZE)
    image = None
    limits = _color_limits(data.blocks, vmin=vmin, vmax=vmax)
    for block in sorted(data.blocks, key=lambda item: (item.level or 0, item.patch or 0)):
        extent = _extent(block)
        image = axes.imshow(
            _display_data(block.data, fill_value=limits[0]),
            origin="lower",
            extent=extent,
            aspect="auto",
            cmap=cmap,
            vmin=limits[0],
            vmax=limits[1],
            interpolation="nearest",
        )
        if show_mesh:
            x0, x1, y0, y1 = extent
            axes.add_patch(
                Rectangle(
                    (x0, y0),
                    x1 - x0,
                    y1 - y0,
                    fill=False,
                    edgecolor=mesh_color,
                    linewidth=mesh_linewidth,
                    alpha=0.9,
                )
            )
    if image is None:
        raise ValueError("SliceData contains no blocks to plot.")

    x_axis, y_axis = _plot_axes(data.blocks[0])
    axes.set_xlabel(x_axis)
    axes.set_ylabel(y_axis)
    axes.set_title(title or _slice_title(data))
    figure.colorbar(image, ax=axes, label=field_label(data.field.name, data.field.units))

    saved = save_figure(figure, output)
    return PlotResult(figure=figure, axes=axes, output=saved)


def _extent(block: GridBlock) -> tuple[float, float, float, float]:
    y_count, x_count = block.data.shape
    y0, x0 = block.origin
    dy, dx = block.spacing
    return (x0, x0 + dx * x_count, y0, y0 + dy * y_count)


def _plot_axes(block: GridBlock) -> tuple[str, str]:
    if len(block.axes) >= 2:
        return block.axes[1], block.axes[0]
    return "x", "y"


def _color_limits(
    blocks: tuple[GridBlock, ...],
    *,
    vmin: float | None,
    vmax: float | None,
) -> tuple[float | None, float | None]:
    if vmin is not None or vmax is not None:
        return vmin, vmax
    arrays = [np.asarray(block.data, dtype=float).ravel() for block in blocks]
    finite = np.concatenate(arrays)
    finite = finite[np.isfinite(finite)]
    finite = finite[np.abs(finite) < DEFAULT_EXTREME_VALUE_LIMIT]
    if finite.size == 0:
        return None, None
    return float(np.nanpercentile(finite, 1)), float(np.nanpercentile(finite, 99))


def _display_data(data: np.ndarray, *, fill_value: float | None):
    values = np.array(data, dtype=float, copy=True)
    valid = np.isfinite(values) & (np.abs(values) < DEFAULT_EXTREME_VALUE_LIMIT)
    values[~valid] = 0.0 if fill_value is None else fill_value
    return values


def _slice_title(data: SliceData) -> str:
    if data.time is None:
        return f"{data.field.name} {data.plane}, iteration {data.iteration}"
    return f"{data.field.name} {data.plane}, iteration {data.iteration}, t={data.time:g}"
