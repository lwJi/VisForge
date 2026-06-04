"""Shared backend interface."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from visforge.data.model import FieldData, FieldInfo, LineData, SliceData


class DataBackend(Protocol):
    """Minimal interface implemented by concrete data readers."""

    path: Path

    def list_iterations(self) -> tuple[int, ...]:
        """Return available iterations."""

    def list_fields(self, iteration: int | None = None) -> tuple[FieldInfo, ...]:
        """Return fields available at an iteration, or globally if possible."""

    def read_slice(
        self,
        field: str,
        *,
        iteration: int | None = None,
        plane: str | None = None,
    ) -> SliceData:
        """Read a 2D scalar slice or native plane output."""

    def read_field(
        self,
        field: str,
        *,
        iteration: int | None = None,
    ) -> FieldData:
        """Read a 3D scalar field."""

    def read_line(
        self,
        field: str,
        *,
        iteration: int | None = None,
        axis: str | None = None,
    ) -> LineData:
        """Read a 1D line output."""


class BackendError(RuntimeError):
    """Base exception for backend failures."""


class BackendUnavailableError(BackendError):
    """Raised when optional backend dependencies are not installed."""


class UnsupportedOperationError(BackendError):
    """Raised when a backend does not support the requested operation."""
