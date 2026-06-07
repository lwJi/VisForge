"""Figure output helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

DEFAULT_DPI = 160


def save_figure(figure: Any, output: str | Path | None, *, dpi: int = DEFAULT_DPI) -> Path | None:
    """Save a Matplotlib figure if an output path was provided."""

    if output is None:
        return None
    if dpi <= 0:
        raise ValueError("Figure output dpi must be positive.")
    path = Path(output).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    return path
