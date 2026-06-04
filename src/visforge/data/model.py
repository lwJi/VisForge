"""Backend-neutral data structures used by loaders, workflows, and plots."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

BackendName = Literal["openpmd", "silo", "tsv"]


@dataclass(frozen=True)
class DataFile:
    """One discovered data artifact in a CarpetX output tree."""

    path: Path
    backend: BackendName
    iteration: int | None = None
    plane: str | None = None
    axis: str | None = None
    variable: str | None = None
    role: str = "data"


@dataclass(frozen=True)
class DatasetIndex:
    """Normalized view of a CarpetX dataset directory or single data file."""

    root: Path
    files: tuple[DataFile, ...]
    manifests: tuple[Path, ...] = ()
    metadata: tuple[Path, ...] = ()

    @property
    def backends(self) -> tuple[BackendName, ...]:
        return tuple(sorted({file.backend for file in self.files}))

    @property
    def iterations(self) -> tuple[int, ...]:
        return tuple(sorted({file.iteration for file in self.files if file.iteration is not None}))

    @property
    def planes(self) -> tuple[str, ...]:
        return tuple(sorted({file.plane for file in self.files if file.plane is not None}))

    @property
    def axes(self) -> tuple[str, ...]:
        return tuple(sorted({file.axis for file in self.files if file.axis is not None}))

    def by_backend(self, backend: BackendName) -> tuple[DataFile, ...]:
        return tuple(file for file in self.files if file.backend == backend)


@dataclass(frozen=True)
class FieldInfo:
    """Description of a field available from a backend."""

    name: str
    components: tuple[str, ...] = ()
    dimensions: int | None = None
    centering: str | None = None
    units: str | None = None
    shape: tuple[int, ...] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GridBlock:
    """One rectangular block of field data."""

    data: NDArray[np.floating[Any]]
    axes: tuple[str, ...]
    origin: tuple[float, ...]
    spacing: tuple[float, ...]
    level: int | None = None
    patch: int | None = None
    ghost_zones: tuple[int, ...] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FieldData:
    """A possibly multi-block scalar field at one iteration."""

    field: FieldInfo
    iteration: int
    time: float | None
    blocks: tuple[GridBlock, ...]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PlaneSpec:
    """User-defined 2D sampling plane embedded in 3D coordinates."""

    origin: tuple[float, float, float]
    normal: tuple[float, float, float]
    up: tuple[float, float, float]
    size: tuple[float, float]
    resolution: tuple[int, int]
    interpolation: Literal["linear", "nearest"] = "linear"


@dataclass(frozen=True)
class SliceData:
    """A 2D scalar slice or native 2D plane output."""

    field: FieldInfo
    iteration: int
    time: float | None
    plane: str
    blocks: tuple[GridBlock, ...]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LineData:
    """One or more line-output samples for a scalar field."""

    field: FieldInfo
    iteration: int
    time: float | None
    axis: str
    coordinate: NDArray[np.floating[Any]]
    values: NDArray[np.floating[Any]]
    patch: NDArray[np.integer[Any]] | None = None
    level: NDArray[np.integer[Any]] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DatasetSummary:
    """Human-readable summary returned by the inspect workflow."""

    root: Path
    backends: tuple[str, ...]
    iterations: tuple[int, ...]
    planes: tuple[str, ...]
    axes: tuple[str, ...]
    fields: tuple[str, ...]
    file_count: int
    metadata_count: int
