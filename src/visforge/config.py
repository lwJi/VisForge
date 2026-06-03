"""YAML configuration helpers for VisForge workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path | None) -> dict[str, Any]:
    """Load a YAML config file, returning an empty mapping when omitted."""

    if path is None:
        return {}
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping at top level of config file {config_path}")
    return data


def section(config: dict[str, Any], name: str) -> dict[str, Any]:
    """Return a named config section as a mapping."""

    value = config.get(name, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"Expected config section {name!r} to be a mapping.")
    return value


def range_value(value: Any) -> tuple[float, float] | None:
    """Parse a two-element numeric range from config data."""

    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError("Expected range to contain exactly two numeric values.")
    return float(value[0]), float(value[1])


def bool_value(value: Any) -> bool:
    """Parse booleans from YAML values."""

    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return bool(value)
