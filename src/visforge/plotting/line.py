"""Line plotting for backend-neutral ``LineData`` objects."""

from __future__ import annotations

from pathlib import Path

from visforge.data.model import LineData
from visforge.plotting.base import PlotResult
from visforge.plotting.output import save_figure
from visforge.plotting.style import LINE_FIGSIZE, configure_matplotlib_environment, field_label


def plot_line(
    line: LineData,
    *,
    output: str | Path | None = None,
    title: str | None = None,
    color: str = "tab:blue",
) -> PlotResult:
    """Plot a scalar line output."""

    configure_matplotlib_environment()
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(figsize=LINE_FIGSIZE)
    axes.plot(line.coordinate, line.values, color=color, linewidth=1.5)
    axes.set_xlabel(line.axis)
    axes.set_ylabel(field_label(line.field.name, line.field.units))
    axes.grid(True, color="0.85", linewidth=0.8)
    axes.set_title(title or _line_title(line))

    saved = save_figure(figure, output)
    return PlotResult(figure=figure, axes=axes, output=saved)


def _line_title(line: LineData) -> str:
    if line.time is None:
        return f"{line.field.name} along {line.axis}, iteration {line.iteration}"
    return f"{line.field.name} along {line.axis}, iteration {line.iteration}, t={line.time:g}"
