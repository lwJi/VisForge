"""Silo backend scaffold.

The interface is present from the start, but actual Silo reads require a Python
binding that is not available in the current environment.
"""

from __future__ import annotations

from pathlib import Path

from visforge.data.backend import BackendUnavailableError
from visforge.data.carpetx import discover
from visforge.data.model import FieldInfo


class SiloBackend:
    """Discover Silo artifacts and fail clearly for unsupported reads."""

    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser().resolve()
        self.index = discover(self.path)
        self._files = self.index.by_backend("silo")
        if not self._files:
            raise FileNotFoundError(f"No CarpetX Silo files found under {self.path}")

    def list_iterations(self) -> tuple[int, ...]:
        return tuple(sorted({file.iteration for file in self._files if file.iteration is not None}))

    def list_fields(self, iteration: int | None = None) -> tuple[FieldInfo, ...]:
        _raise_unavailable()

    def read_slice(self, field: str, *, iteration: int | None = None, plane: str | None = None):
        _raise_unavailable()

    def read_line(self, field: str, *, iteration: int | None = None, axis: str | None = None):
        _raise_unavailable()


def _raise_unavailable():
    raise BackendUnavailableError(
        "Silo reading requires a Python Silo binding, but none is installed in this environment."
    )
