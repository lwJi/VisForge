"""Central plotting defaults."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

DEFAULT_CMAP = "viridis"
COLORMAP_SUGGESTIONS = ("viridis", "plasma", "inferno", "magma", "cividis")
DEFAULT_FIGSIZE = (7.0, 5.0)
LINE_FIGSIZE = (7.0, 4.0)
PLOT_FONT_FAMILY = "STIXGeneral"
AXIS_LABELS = {
    "x": r"$x$",
    "y": r"$y$",
    "z": r"$z$",
}


def configure_matplotlib_environment() -> None:
    """Point Matplotlib/font caches at a writable temp directory when needed."""

    cache_root = Path(tempfile.gettempdir()) / "visforge-matplotlib-cache"
    mpl_config = cache_root / "mplconfig"
    xdg_cache = cache_root / "xdg"
    mpl_config.mkdir(parents=True, exist_ok=True)
    xdg_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config))
    os.environ.setdefault("XDG_CACHE_HOME", str(xdg_cache))
    os.environ.setdefault("MPLBACKEND", "Agg")


def configure_matplotlib_style() -> None:
    """Apply VisForge's plotting typography defaults."""

    configure_matplotlib_environment()
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [PLOT_FONT_FAMILY],
            "mathtext.fontset": "stix",
            "axes.titlesize": 16,
            "axes.labelsize": 15,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            "figure.titlesize": 16,
        }
    )


def normalize_colormap(name: str) -> str:
    """Return a valid Matplotlib colormap name or raise a concise user error."""

    normalized = str(name).strip()
    if not normalized:
        raise ValueError("Colormap name cannot be empty.")
    configure_matplotlib_environment()
    import matplotlib as mpl

    try:
        mpl.colormaps[normalized]
    except KeyError:
        suggestions = ", ".join(COLORMAP_SUGGESTIONS)
        raise ValueError(f"Unknown colormap '{name}'. Try one of: {suggestions}.") from None
    return normalized


def field_label(name: str, units: str | None = None) -> str:
    label = latex_identifier(name)
    if units:
        return rf"${label}\,[{latex_identifier(units)}]$"
    return rf"${label}$"


def axis_label(name: str) -> str:
    return AXIS_LABELS.get(name, rf"${latex_identifier(name)}$")


def plane_label(name: str) -> str:
    if len(name) == 2 and all(axis in AXIS_LABELS for axis in name):
        return rf"${name}$"
    return rf"${latex_identifier(name)}$"


def latex_identifier(value: str) -> str:
    escaped = (
        value.replace("\\", r"\backslash ")
        .replace("_", r"\_")
        .replace("%", r"\%")
        .replace("#", r"\#")
        .replace("&", r"\&")
    )
    return rf"\mathrm{{{escaped}}}"
