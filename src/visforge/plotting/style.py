"""Central plotting defaults."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

DEFAULT_CMAP = "viridis"
DEFAULT_FIGSIZE = (7.0, 5.0)
LINE_FIGSIZE = (7.0, 4.0)


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


def field_label(name: str, units: str | None = None) -> str:
    if units:
        return f"{name} [{units}]"
    return name
