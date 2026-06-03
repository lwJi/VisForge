"""openPMD backend for CarpetX `.bp5` output."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np

from visforge.data.backend import BackendUnavailableError, UnsupportedOperationError
from visforge.data.carpetx import discover
from visforge.data.model import FieldInfo, GridBlock, SliceData

try:  # pragma: no cover - exercised when dependency is absent
    import openpmd_api as io
except ImportError:  # pragma: no cover
    io = None

PATCH_LEVEL_RE = re.compile(r"^(?P<base>.+)_patch(?P<patch>\d+)_lev(?P<level>\d+)$")


class OpenPMDBackend:
    """Read scalar CarpetX fields from openPMD BP5 files."""

    def __init__(self, path: str | Path):
        if io is None:
            raise BackendUnavailableError("openPMD-api is not installed.")
        self.path = Path(path).expanduser().resolve()
        self.index = discover(self.path)
        self._files = self.index.by_backend("openpmd")
        if not self._files:
            raise FileNotFoundError(f"No CarpetX openPMD .bp5 files found under {self.path}")

    def list_iterations(self) -> tuple[int, ...]:
        return tuple(sorted({file.iteration for file in self._files if file.iteration is not None}))

    def list_fields(self, iteration: int | None = None) -> tuple[FieldInfo, ...]:
        files = self._select_files(iteration=iteration)
        fields: dict[str, FieldInfo] = {}
        for file in files:
            series = io.Series(str(file.path), io.Access.read_only)
            try:
                opmd_iteration = _first_iteration(series)
                for mesh_name in opmd_iteration.meshes:
                    short = _short_field_name(mesh_name)
                    fields.setdefault(
                        short,
                        FieldInfo(
                            name=short,
                            dimensions=None,
                            metadata={"openpmd_mesh_example": mesh_name},
                        ),
                    )
            finally:
                series.close()
        return tuple(fields[name] for name in sorted(fields))

    def read_slice(
        self,
        field: str,
        *,
        iteration: int | None = None,
        plane: str | None = None,
    ) -> SliceData:
        file = self._select_one_file(iteration=iteration, plane=plane)
        series = io.Series(str(file.path), io.Access.read_only)
        try:
            opmd_iteration = _first_iteration(series)
            time = _attribute(opmd_iteration, "time")
            blocks: list[GridBlock] = []
            for mesh_name in opmd_iteration.meshes:
                if not _matches_field(mesh_name, field):
                    continue
                mesh = opmd_iteration.meshes[mesh_name]
                component_name = _component_name(mesh, field)
                component = mesh[component_name]
                array = np.asarray(component.load_chunk(), dtype=float)
                series.flush()
                block = _to_slice_block(
                    array,
                    mesh,
                    mesh_name=mesh_name,
                    requested_plane=plane or file.plane,
                )
                blocks.append(block)

            if not blocks:
                raise ValueError(f"Field {field!r} was not found in {file.path}")

            inferred_plane = plane or file.plane or _plane_from_axes(blocks[0].axes)
            return SliceData(
                field=FieldInfo(name=field, dimensions=2),
                iteration=file.iteration if file.iteration is not None else int(_first_iteration_key(series)),
                time=float(time) if time is not None else None,
                plane=inferred_plane,
                blocks=tuple(blocks),
                metadata={"source": str(file.path)},
            )
        finally:
            series.close()

    def read_line(self, field: str, *, iteration: int | None = None, axis: str | None = None):
        raise UnsupportedOperationError("openPMD line reads are not implemented yet; use TSV outputs.")

    def _select_files(self, *, iteration: int | None = None, plane: str | None = None):
        files = self._files
        if iteration is not None:
            files = tuple(file for file in files if file.iteration == iteration)
        if plane is not None:
            files = tuple(file for file in files if file.plane == plane)
        return files

    def _select_one_file(self, *, iteration: int | None, plane: str | None):
        files = self._select_files(iteration=iteration, plane=plane)
        if iteration is None and files:
            latest = max(file.iteration or 0 for file in files)
            files = tuple(file for file in files if (file.iteration or 0) == latest)
        if plane is None:
            non_plane = tuple(file for file in files if file.plane is None)
            if len(non_plane) == 1:
                return non_plane[0]
            plane_files = tuple(file for file in files if file.plane is not None)
            if len(plane_files) == 1:
                return plane_files[0]
        if len(files) == 1:
            return files[0]
        if not files:
            raise FileNotFoundError("No openPMD file matches the requested iteration and plane.")
        names = ", ".join(file.path.name for file in files[:8])
        raise ValueError(f"openPMD selection is ambiguous; specify --plane. Matches include: {names}")


def _first_iteration(series: Any):
    return series.iterations[_first_iteration_key(series)]


def _first_iteration_key(series: Any) -> int:
    return list(series.iterations)[0]


def _attribute(record: Any, name: str) -> Any:
    try:
        return record.get_attribute(name)
    except Exception:
        return None


def _short_field_name(mesh_name: str) -> str:
    base = _mesh_base(mesh_name)
    return base.rsplit("_", 1)[-1]


def _mesh_base(mesh_name: str) -> str:
    match = PATCH_LEVEL_RE.match(mesh_name)
    if match is None:
        return mesh_name
    return match.group("base")


def _matches_field(mesh_name: str, field: str) -> bool:
    base = _mesh_base(mesh_name)
    return field in {base, base.rsplit("_", 1)[-1]}


def _component_name(mesh: Any, field: str) -> str:
    components = list(mesh)
    if not components:
        raise ValueError("openPMD mesh has no record components.")
    for component in components:
        if component == field or component.endswith(f"_{field}"):
            return component
    return components[0]


def _to_slice_block(
    array: np.ndarray,
    mesh: Any,
    *,
    mesh_name: str,
    requested_plane: str | None,
) -> GridBlock:
    axis_labels = tuple(str(axis) for axis in getattr(mesh, "axis_labels", ()))
    spacing = tuple(float(value) for value in getattr(mesh, "grid_spacing", (1.0,) * array.ndim))
    origin = tuple(float(value) for value in getattr(mesh, "grid_global_offset", (0.0,) * array.ndim))

    data, axes, block_origin, block_spacing = _reduce_to_2d(
        array,
        axis_labels=axis_labels,
        origin=origin,
        spacing=spacing,
        requested_plane=requested_plane,
    )
    patch, level = _patch_and_level(mesh_name)
    return GridBlock(
        data=data,
        axes=axes,
        origin=block_origin,
        spacing=block_spacing,
        patch=patch,
        level=level,
        metadata={"openpmd_mesh": mesh_name},
    )


def _reduce_to_2d(
    array: np.ndarray,
    *,
    axis_labels: tuple[str, ...],
    origin: tuple[float, ...],
    spacing: tuple[float, ...],
    requested_plane: str | None,
) -> tuple[np.ndarray, tuple[str, ...], tuple[float, ...], tuple[float, ...]]:
    axes = axis_labels or tuple(f"axis{i}" for i in range(array.ndim))
    data = np.squeeze(array)
    if data.ndim == 2:
        kept = [i for i, size in enumerate(array.shape) if size != 1]
        if len(kept) != 2:
            kept = list(range(len(axes)))[-2:]
        return (
            data,
            tuple(axes[i] for i in kept),
            tuple(origin[i] for i in kept),
            tuple(spacing[i] for i in kept),
        )

    if array.ndim != 3:
        raise ValueError(f"Expected 2D or 3D openPMD data, got shape {array.shape}.")

    normal_axis = _normal_axis(requested_plane)
    normal_index = axes.index(normal_axis) if normal_axis in axes else 0
    center = array.shape[normal_index] // 2
    data = np.take(array, center, axis=normal_index)
    kept = [i for i in range(3) if i != normal_index]
    return (
        np.asarray(data),
        tuple(axes[i] for i in kept),
        tuple(origin[i] for i in kept),
        tuple(spacing[i] for i in kept),
    )


def _normal_axis(plane: str | None) -> str:
    if plane == "xy":
        return "z"
    if plane == "xz":
        return "y"
    if plane == "yz":
        return "x"
    return "z"


def _plane_from_axes(axes: tuple[str, ...]) -> str:
    joined = "".join(axis for axis in ("x", "y", "z") if axis in axes)
    return joined or "unknown"


def _patch_and_level(mesh_name: str) -> tuple[int | None, int | None]:
    match = PATCH_LEVEL_RE.match(mesh_name)
    if match is None:
        return None, None
    return int(match.group("patch")), int(match.group("level"))
