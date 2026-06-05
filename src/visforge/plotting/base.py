"""Shared plotting return types."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from string import Formatter
from typing import Any

TITLE_PLACEHOLDER_RE = re.compile(r"\{(?P<name>[A-Za-z_]\w*)(?::(?P<format>[^{}]*))?\}")


@dataclass(frozen=True)
class PlotLabels:
    """User-provided plot text overrides."""

    title: str | None = None
    xlabel: str | None = None
    ylabel: str | None = None
    legend: str | None = None
    colorbar: str | None = None


@dataclass(frozen=True)
class PlotResult:
    """Result of creating a plot."""

    figure: Any
    axes: Any
    output: Path | None = None


def format_title_template(template: str | None, **values: Any) -> str | None:
    """Substitute known title placeholders while leaving other braces alone."""

    if template is None:
        return None

    def replace_match(match: re.Match[str]) -> str:
        name = match.group("name")
        if name not in values:
            return match.group(0)
        value = values[name]
        if value is None:
            return ""
        format_spec = match.group("format") or ""
        try:
            return Formatter().format_field(value, format_spec)
        except (TypeError, ValueError) as exc:
            placeholder = match.group(0)
            raise ValueError(f"Invalid title placeholder {placeholder}: {exc}") from None

    return TITLE_PLACEHOLDER_RE.sub(replace_match, template)
