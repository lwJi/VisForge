"""Line plotting for backend-neutral ``LineData`` objects."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from visforge.data.model import LineData
from visforge.plotting.base import PlotLabels, PlotResult, format_title_template
from visforge.plotting.output import save_figure
from visforge.plotting.style import (
    LINE_FIGSIZE,
    axis_label,
    configure_matplotlib_style,
    field_label,
)


def plot_line(
    line: LineData,
    *,
    output: str | Path | None = None,
    labels: PlotLabels | None = None,
    title: str | None = None,
    color: str = "tab:blue",
) -> PlotResult:
    """Plot a scalar line output."""

    configure_matplotlib_style()
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(figsize=LINE_FIGSIZE)
    labels = _resolve_labels(labels, title=title)
    axes.plot(line.coordinate, line.values, color=color, linewidth=1.5, label=labels.legend)
    axes.set_xlabel(labels.xlabel or axis_label(line.axis))
    axes.set_ylabel(labels.ylabel or field_label(line.field.name, line.field.units))
    axes.grid(True, color="0.85", linewidth=0.8)
    axes.set_title(_resolve_title(line, labels.title) or _line_title(line))
    if labels.legend is not None:
        axes.legend()

    saved = save_figure(figure, output)
    return PlotResult(figure=figure, axes=axes, output=saved)


def _line_title(line: LineData) -> str:
    label = field_label(line.field.name, line.field.units)
    axis = axis_label(line.axis)
    if line.time is None:
        return f"{label} along {axis}, iteration {line.iteration}"
    return f"{label} along {axis}, iteration {line.iteration}, $t={line.time:g}$"


def _resolve_title(line: LineData, title: str | None) -> str | None:
    return format_title_template(
        title,
        field=line.field.name,
        units=line.field.units,
        iteration=line.iteration,
        time=line.time,
        axis=line.axis,
    )


def _resolve_labels(labels: PlotLabels | None, *, title: str | None) -> PlotLabels:
    if labels is None:
        return PlotLabels(title=title)
    if labels.title is None and title is not None:
        return replace(labels, title=title)
    return labels
