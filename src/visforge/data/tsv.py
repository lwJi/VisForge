"""Backend for CarpetX TSV line-output files."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from visforge.data.backend import UnsupportedOperationError
from visforge.data.carpetx import discover
from visforge.data.detect import parse_tsv_name
from visforge.data.model import FieldInfo, LineData


class TsvBackend:
    """Read CarpetX line-output TSV files."""

    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser().resolve()
        self.index = discover(self.path)
        self._files = self.index.by_backend("tsv")
        if not self._files:
            raise FileNotFoundError(f"No CarpetX TSV line-output files found under {self.path}")

    def list_iterations(self) -> tuple[int, ...]:
        return tuple(sorted({file.iteration for file in self._files if file.iteration is not None}))

    def list_fields(self, iteration: int | None = None) -> tuple[FieldInfo, ...]:
        files = self._filter(iteration=iteration)
        names = sorted({file.variable for file in files if file.variable is not None})
        return tuple(FieldInfo(name=name, dimensions=1) for name in names)

    def read_slice(self, field: str, *, iteration: int | None = None, plane: str | None = None):
        raise UnsupportedOperationError("TSV files contain line outputs, not 2D slices.")

    def read_line(
        self,
        field: str,
        *,
        iteration: int | None = None,
        axis: str | None = None,
    ) -> LineData:
        matches = self._filter(field=field, iteration=iteration, axis=axis)
        if not matches:
            raise FileNotFoundError(_missing_message(field, iteration, axis, self.path))
        if len(matches) > 1:
            available = ", ".join(str(file.path.name) for file in matches)
            raise ValueError(f"Line selection is ambiguous; matches: {available}")

        file = matches[0]
        columns, values = _read_tsv(file.path)
        axis_name = file.axis or axis
        if axis_name is None:
            raise ValueError("Axis is required when it cannot be inferred from the TSV filename.")
        if axis_name not in columns:
            raise ValueError(f"TSV file {file.path} does not contain coordinate column {axis_name!r}.")
        if field not in columns:
            raise ValueError(f"TSV file {file.path} does not contain field column {field!r}.")

        iteration_index = columns.index("iteration") if "iteration" in columns else None
        time_index = columns.index("time") if "time" in columns else None
        patch_index = columns.index("patch") if "patch" in columns else None
        level_index = columns.index("level") if "level" in columns else None

        coordinate = values[:, columns.index(axis_name)].astype(float)
        field_values = values[:, columns.index(field)].astype(float)
        order = np.argsort(coordinate)

        actual_iteration = file.iteration
        if iteration_index is not None and values.size:
            actual_iteration = int(values[0, iteration_index])

        time = None
        if time_index is not None and values.size:
            time = float(values[0, time_index])

        patch = None
        if patch_index is not None:
            patch = values[:, patch_index].astype(int)[order]

        level = None
        if level_index is not None:
            level = values[:, level_index].astype(int)[order]

        return LineData(
            field=FieldInfo(name=field, dimensions=1),
            iteration=actual_iteration if actual_iteration is not None else 0,
            time=time,
            axis=axis_name,
            coordinate=coordinate[order],
            values=field_values[order],
            patch=patch,
            level=level,
            metadata={"source": str(file.path), "columns": tuple(columns)},
        )

    def _filter(
        self,
        *,
        field: str | None = None,
        iteration: int | None = None,
        axis: str | None = None,
    ):
        files = self._files
        if field is not None:
            files = tuple(file for file in files if file.variable == field)
        if iteration is not None:
            files = tuple(file for file in files if file.iteration == iteration)
        if axis is not None:
            files = tuple(file for file in files if file.axis == axis)
        return files


def _read_tsv(path: Path) -> tuple[list[str], np.ndarray]:
    with path.open("r", encoding="utf-8") as handle:
        header = handle.readline().strip()
    columns = _parse_header(header)
    values = np.loadtxt(path, comments="#", ndmin=2)
    return columns, values


def _parse_header(header: str) -> list[str]:
    if header.startswith("#"):
        header = header[1:].strip()
    columns: list[str] = []
    for part in header.split("\t"):
        if ":" not in part:
            continue
        columns.append(part.split(":", 1)[1].strip())
    if not columns:
        raise ValueError("TSV header does not contain numbered CarpetX columns.")
    return columns


def _missing_message(field: str, iteration: int | None, axis: str | None, path: Path) -> str:
    requested = [f"field={field!r}"]
    if iteration is not None:
        requested.append(f"iteration={iteration}")
    if axis is not None:
        requested.append(f"axis={axis!r}")
    return f"No TSV line output matching {', '.join(requested)} under {path}"


def is_tsv_line_file(path: str | Path) -> bool:
    return parse_tsv_name(Path(path)) is not None
