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
    axis_label,
    configure_matplotlib_style,
    field_label,
    plane_label,
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
    mesh_linewidth: float = 0.15,
    mesh_alpha: float = 0.75,
    mesh_max_lines: int | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
) -> PlotResult:
    """Plot one or more blocks from a scalar 2D slice."""

    configure_matplotlib_style()
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    figure, axes = plt.subplots(figsize=DEFAULT_FIGSIZE)
    image = None
    plotted_blocks = tuple(sorted(data.blocks, key=lambda item: (item.level or 0, item.patch or 0)))
    limits = _color_limits(data.blocks, vmin=vmin, vmax=vmax)
    norm = Normalize(vmin=limits[0], vmax=limits[1])
    for block in plotted_blocks:
        block_data, extent = _valid_data_and_extent(block)
        if block_data.size == 0:
            continue
        image = axes.imshow(
            _display_data(block_data, fill_value=limits[0]),
            origin="lower",
            extent=extent,
            aspect="equal",
            cmap=cmap,
            norm=norm,
            interpolation="nearest",
        )
    if image is None:
        raise ValueError("SliceData contains no blocks to plot.")
    if show_mesh:
        _draw_mesh_overlay(
            axes,
            plotted_blocks,
            color=mesh_color,
            linewidth=mesh_linewidth,
            alpha=mesh_alpha,
            max_lines=mesh_max_lines,
        )

    x_axis, y_axis = _plot_axes(data.blocks[0])
    axes.set_xlabel(axis_label(x_axis))
    axes.set_ylabel(axis_label(y_axis))
    if xlim is not None:
        axes.set_xlim(xlim)
    if ylim is not None:
        axes.set_ylim(ylim)
    axes.set_aspect("equal", adjustable="box")
    axes.set_title(title or _slice_title(data))
    figure.colorbar(image, ax=axes, label=field_label(data.field.name, data.field.units))

    saved = save_figure(figure, output)
    return PlotResult(figure=figure, axes=axes, output=saved)


def _extent(block: GridBlock) -> tuple[float, float, float, float]:
    y_count, x_count = block.data.shape
    y0, x0 = block.origin
    dy, dx = block.spacing
    y_position, x_position = _grid_position(block)
    return (
        *_axis_extent(x0, dx, 0, x_count, x_position),
        *_axis_extent(y0, dy, 0, y_count, y_position),
    )


def _valid_data_and_extent(block: GridBlock) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    extent = _mesh_extent(block)
    y_slice, x_slice = _extent_slices(block, extent)
    data = np.asarray(block.data)[y_slice, x_slice]
    x0, _, y0, _ = extent
    dy, dx = block.spacing
    x_start = x_slice.start or 0
    y_start = y_slice.start or 0
    x_stop = x_slice.stop if x_slice.stop is not None else x_start + data.shape[1]
    y_stop = y_slice.stop if y_slice.stop is not None else y_start + data.shape[0]
    block_y0, block_x0 = block.origin
    block_dy, block_dx = block.spacing
    block_y_position, block_x_position = _grid_position(block)
    cropped_extent = (
        *_axis_extent(block_x0, block_dx, x_start, x_stop, block_x_position),
        *_axis_extent(block_y0, block_dy, y_start, y_stop, block_y_position),
    )
    if data.size == 0:
        return data, extent
    return data, _clip_extent(cropped_extent, extent)


def _extent_slices(
    block: GridBlock,
    extent: tuple[float, float, float, float],
) -> tuple[slice, slice]:
    y_count, x_count = block.data.shape
    y0, x0 = block.origin
    dy, dx = block.spacing
    ex0, ex1, ey0, ey1 = extent
    y_position, x_position = _grid_position(block)
    x_start, x_stop = _axis_extent_slice(
        origin=x0,
        spacing=dx,
        count=x_count,
        position=x_position,
        lower=ex0,
        upper=ex1,
    )
    y_start, y_stop = _axis_extent_slice(
        origin=y0,
        spacing=dy,
        count=y_count,
        position=y_position,
        lower=ey0,
        upper=ey1,
    )
    return slice(y_start, y_stop), slice(x_start, x_stop)


def _grid_position(block: GridBlock) -> tuple[float, float]:
    position = block.metadata.get("grid_position")
    if position is None:
        return (0.5, 0.5)
    if len(position) < 2:
        return (0.5, 0.5)
    return float(position[0]), float(position[1])


def _axis_extent_slice(
    *,
    origin: float,
    spacing: float,
    count: int,
    position: float,
    lower: float,
    upper: float,
) -> tuple[int, int]:
    point_origin = origin + position * spacing
    start = int(np.ceil((lower - point_origin) / spacing))
    stop = int(np.floor((upper - point_origin) / spacing)) + 1
    return max(0, start), min(count, stop)


def _axis_extent(
    origin: float,
    spacing: float,
    start: int,
    stop: int,
    position: float,
) -> tuple[float, float]:
    if stop <= start:
        edge = origin + (start + position) * spacing
        return edge, edge
    first = origin + (start + position) * spacing
    last = origin + (stop - 1 + position) * spacing
    if np.isclose(position, 0.0) or np.isclose(position, 1.0):
        return first, last
    return first - 0.5 * spacing, last + 0.5 * spacing


def _clip_extent(
    candidate: tuple[float, float, float, float],
    target: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    x0, x1, y0, y1 = candidate
    tx0, tx1, ty0, ty1 = target
    return max(x0, tx0), min(x1, tx1), max(y0, ty0), min(y1, ty1)


def _plot_axes(block: GridBlock) -> tuple[str, str]:
    if len(block.axes) >= 2:
        return block.axes[1], block.axes[0]
    return "x", "y"


def _draw_mesh_overlay(
    axes,
    blocks: tuple[GridBlock, ...],
    *,
    color: str,
    linewidth: float,
    alpha: float,
    max_lines: int | None,
) -> None:
    from matplotlib.patches import Rectangle

    alpha = _clamp_alpha(alpha)
    for block in blocks:
        x0, x1, y0, y1 = _mesh_extent(block)
        if x1 <= x0 or y1 <= y0:
            continue
        rectangle = Rectangle(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            fill=False,
            edgecolor=color,
            linewidth=linewidth,
            alpha=alpha,
            zorder=20,
        )
        axes.add_patch(rectangle)

        xs, ys = _mesh_line_positions(block, extent=(x0, x1, y0, y1), max_lines=max_lines)
        vlines = axes.vlines(
            xs,
            y0,
            y1,
            colors=color,
            linewidth=linewidth,
            alpha=alpha,
            zorder=19,
        )
        hlines = axes.hlines(
            ys,
            x0,
            x1,
            colors=color,
            linewidth=linewidth,
            alpha=alpha,
            zorder=19,
        )


def _clamp_alpha(alpha: float) -> float:
    return min(1.0, max(0.0, float(alpha)))


def _mesh_extent(block: GridBlock) -> tuple[float, float, float, float]:
    extent = block.metadata.get("refinement_extent")
    if extent is None:
        return _extent(block)
    return tuple(float(value) for value in extent)


def _mesh_line_positions(
    block: GridBlock,
    *,
    extent: tuple[float, float, float, float],
    max_lines: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    y0, x0 = block.origin
    dy, dx = block.spacing
    ex0, ex1, ey0, ey1 = extent
    y_slice, x_slice = _extent_slices(block, extent)
    x_start = x_slice.start or 0
    x_stop = x_slice.stop or x_start
    y_start = y_slice.start or 0
    y_stop = y_slice.stop or y_start
    if max_lines is None or max_lines <= 0:
        x_step = 1
        y_step = 1
    else:
        x_step = max(1, int(np.ceil(max(1, x_stop - x_start) / max_lines)))
        y_step = max(1, int(np.ceil(max(1, y_stop - y_start) / max_lines)))
    xs = x0 + dx * np.arange(x_start, x_stop + 1, x_step)
    ys = y0 + dy * np.arange(y_start, y_stop + 1, y_step)
    return _ensure_endpoint(xs, ex1), _ensure_endpoint(ys, ey1)


def _ensure_endpoint(values: np.ndarray, endpoint: float) -> np.ndarray:
    if values.size == 0 or not np.isclose(values[-1], endpoint):
        return np.append(values, endpoint)
    return values


def _color_limits(
    blocks: tuple[GridBlock, ...],
    *,
    vmin: float | None,
    vmax: float | None,
) -> tuple[float | None, float | None]:
    arrays = [np.asarray(_valid_data_and_extent(block)[0], dtype=float).ravel() for block in blocks]
    arrays = [array for array in arrays if array.size]
    if not arrays:
        return _fallback_color_limits(vmin, vmax)
    finite = np.concatenate(arrays)
    finite = finite[np.isfinite(finite)]
    finite = finite[np.abs(finite) < DEFAULT_EXTREME_VALUE_LIMIT]
    if finite.size == 0:
        return _fallback_color_limits(vmin, vmax)
    lower = float(np.nanpercentile(finite, 1)) if vmin is None else vmin
    upper = float(np.nanpercentile(finite, 99)) if vmax is None else vmax
    return _expand_degenerate_limits(lower, upper)


def _expand_degenerate_limits(
    vmin: float | None,
    vmax: float | None,
) -> tuple[float | None, float | None]:
    if vmin is None or vmax is None:
        return vmin, vmax
    lower = float(vmin)
    upper = float(vmax)
    if not np.isclose(lower, upper):
        return lower, upper
    half_width = abs(lower) * 0.01 if lower != 0.0 else 0.1
    return lower - half_width, upper + half_width


def _fallback_color_limits(
    vmin: float | None,
    vmax: float | None,
) -> tuple[float | None, float | None]:
    if vmin is None and vmax is None:
        return _expand_degenerate_limits(0.0, 0.0)
    value = vmax if vmin is None else vmin
    return _expand_degenerate_limits(value, value)


def _display_data(data: np.ndarray, *, fill_value: float | None):
    values = np.array(data, dtype=float, copy=True)
    valid = np.isfinite(values) & (np.abs(values) < DEFAULT_EXTREME_VALUE_LIMIT)
    values[~valid] = 0.0 if fill_value is None else fill_value
    return values


def _slice_title(data: SliceData) -> str:
    label = field_label(data.field.name, data.field.units)
    plane = plane_label(data.plane)
    if data.time is None:
        return f"{label} {plane}, iteration {data.iteration}"
    return f"{label} {plane}, iteration {data.iteration}, $t={data.time:g}$"
