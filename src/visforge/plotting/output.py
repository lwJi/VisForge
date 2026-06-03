"""Figure output helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def save_figure(figure: Any, output: str | Path | None, *, dpi: int = 160) -> Path | None:
    """Save a Matplotlib figure if an output path was provided."""

    if output is None:
        return None
    path = Path(output).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    return path
