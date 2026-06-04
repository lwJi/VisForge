"""openPMD backend for CarpetX `.bp5` output."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np

from visforge.data.backend import BackendUnavailableError, UnsupportedOperationError
from visforge.data.carpetx import discover
from visforge.data.model import FieldData, FieldInfo, GridBlock, SliceData

try:  # pragma: no cover - exercised when dependency is absent
    import openpmd_api as io
except ImportError:  # pragma: no cover
    io = None

PATCH_LEVEL_RE = re.compile(r"^(?P<base>.+)_patch(?P<patch>\d+)_lev(?P<level>\d+)$")
BOXINBOX_RADIUS_RE = re.compile(r"BoxInBox::radius_(?P<region>\d+)\s*=\s*\[(?P<values>[^\]]+)\]")
CARPETX_BOUND_RE = re.compile(
    r"CarpetX::(?P<axis>[xyz])(?P<side>min|max)\s*=\s*(?P<value>[+-]?\d+(?:\.\d*)?(?:[eE][+-]?\d+)?)"
)


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
        self._refinement_radii = _read_boxinbox_radii(self.path)
        self._domain_bounds = _read_carpetx_domain_bounds(self.path)
        if not self._refinement_radii:
            for file in self._files:
                self._refinement_radii = _read_boxinbox_radii(file.path)
                if self._refinement_radii:
                    break
        if not self._domain_bounds:
            for file in self._files:
                self._domain_bounds = _read_carpetx_domain_bounds(file.path)
                if self._domain_bounds:
                    break

    def list_iterations(self) -> tuple[int, ...]:
        return tuple(sorted({file.iteration for file in self._files if file.iteration is not None}))

    def list_fields(self, iteration: int | None = None) -> tuple[FieldInfo, ...]:
        files = self._select_files(iteration=iteration)
        components: dict[str, set[str]] = {}
        examples: dict[str, str] = {}
        for file in files:
            series = io.Series(str(file.path), io.Access.read_only)
            try:
                opmd_iteration = _first_iteration(series)
                for mesh_name in opmd_iteration.meshes:
                    short = _short_field_name(mesh_name)
                    mesh = opmd_iteration.meshes[mesh_name]
                    components.setdefault(short, set()).update(str(component) for component in list(mesh))
                    examples.setdefault(short, mesh_name)
            finally:
                series.close()
        return tuple(
            FieldInfo(
                name=name,
                components=tuple(sorted(components[name])),
                dimensions=None,
                metadata={"openpmd_mesh_example": examples[name]},
            )
            for name in sorted(components)
        )

    def read_slice(
        self,
        field: str,
        *,
        component: str | None = None,
        iteration: int | None = None,
        plane: str | None = None,
    ) -> SliceData:
        file = self._select_one_file(iteration=iteration, plane=plane)
        series = io.Series(str(file.path), io.Access.read_only)
        try:
            opmd_iteration = _first_iteration(series)
            time = _attribute(opmd_iteration, "time")
            blocks: list[GridBlock] = []
            selected_component: str | None = None
            for mesh_name in opmd_iteration.meshes:
                if not _matches_field(mesh_name, field):
                    continue
                mesh = opmd_iteration.meshes[mesh_name]
                component_name = _component_name(mesh, field, component)
                selected_component = selected_component or component_name
                record_component = mesh[component_name]
                array = np.asarray(record_component.load_chunk(), dtype=float)
                series.flush()
                block = _to_slice_block(
                    array,
                    mesh,
                    record_component,
                    mesh_name=mesh_name,
                    requested_plane=plane or file.plane,
                    refinement_radii=self._refinement_radii,
                    domain_bounds=self._domain_bounds,
                )
                blocks.append(block)

            if not blocks:
                raise ValueError(f"Field {field!r} was not found in {file.path}")

            inferred_plane = plane or file.plane or _plane_from_axes(blocks[0].axes)
            return SliceData(
                field=FieldInfo(
                    name=selected_component or field,
                    components=((selected_component,) if selected_component is not None else ()),
                    dimensions=2,
                    metadata={"openpmd_record": field},
                ),
                iteration=file.iteration if file.iteration is not None else int(_first_iteration_key(series)),
                time=float(time) if time is not None else None,
                plane=inferred_plane,
                blocks=tuple(blocks),
                metadata={"source": str(file.path)},
            )
        finally:
            series.close()

    def read_field(
        self,
        field: str,
        *,
        component: str | None = None,
        iteration: int | None = None,
    ) -> FieldData:
        file = self._select_one_volume_file(iteration=iteration)
        series = io.Series(str(file.path), io.Access.read_only)
        try:
            opmd_iteration = _first_iteration(series)
            time = _attribute(opmd_iteration, "time")
            blocks: list[GridBlock] = []
            selected_component: str | None = None
            for mesh_name in opmd_iteration.meshes:
                if not _matches_field(mesh_name, field):
                    continue
                mesh = opmd_iteration.meshes[mesh_name]
                component_name = _component_name(mesh, field, component)
                selected_component = selected_component or component_name
                record_component = mesh[component_name]
                array = np.asarray(record_component.load_chunk(), dtype=float)
                series.flush()
                blocks.append(
                    _to_field_block(
                        array,
                        mesh,
                        record_component,
                        mesh_name=mesh_name,
                        refinement_radii=self._refinement_radii,
                        domain_bounds=self._domain_bounds,
                    )
                )

            if not blocks:
                raise ValueError(f"Field {field!r} was not found in {file.path}")

            return FieldData(
                field=FieldInfo(
                    name=selected_component or field,
                    components=((selected_component,) if selected_component is not None else ()),
                    dimensions=3,
                    metadata={"openpmd_record": field},
                ),
                iteration=file.iteration if file.iteration is not None else int(_first_iteration_key(series)),
                time=float(time) if time is not None else None,
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

    def _select_one_volume_file(self, *, iteration: int | None):
        files = tuple(file for file in self._select_files(iteration=iteration) if file.plane is None)
        if iteration is None and files:
            latest = max(file.iteration or 0 for file in files)
            files = tuple(file for file in files if (file.iteration or 0) == latest)
        if len(files) == 1:
            return files[0]
        if not files:
            raise FileNotFoundError("No 3D openPMD file matches the requested iteration.")
        names = ", ".join(file.path.name for file in files[:8])
        raise ValueError(f"openPMD volume selection is ambiguous. Matches include: {names}")


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


def _component_name(mesh: Any, field: str, component: str | None) -> str:
    components = list(mesh)
    if not components:
        raise ValueError("openPMD mesh has no record components.")
    if component is not None:
        matches = _matching_components(components, component)
        if not matches:
            available = ", ".join(str(value) for value in components)
            raise ValueError(
                f"Field {field!r} does not contain component {component!r}. "
                f"Available components: {available}"
            )
        if len(matches) > 1:
            available = ", ".join(str(value) for value in matches)
            raise ValueError(f"Component {component!r} is ambiguous. Matches: {available}")
        return str(matches[0])

    if len(components) == 1:
        return str(components[0])

    matches = _matching_components(components, field)
    if len(matches) == 1:
        return str(matches[0])

    available = ", ".join(str(value) for value in components)
    raise ValueError(
        f"Field {field!r} has multiple components: {available}. "
        "Specify one with --component."
    )


def _matching_components(components: list[Any], name: str) -> list[str]:
    exact = [str(component) for component in components if str(component) == name]
    if exact:
        return exact
    return [str(component) for component in components if str(component).endswith(f"_{name}")]


def _to_slice_block(
    array: np.ndarray,
    mesh: Any,
    record_component: Any | None = None,
    *,
    mesh_name: str,
    requested_plane: str | None,
    refinement_radii: tuple[float, ...],
    domain_bounds: dict[str, tuple[float, float]],
) -> GridBlock:
    axis_labels = tuple(str(axis) for axis in getattr(mesh, "axis_labels", ()))
    spacing = tuple(float(value) for value in getattr(mesh, "grid_spacing", (1.0,) * array.ndim))
    origin = tuple(float(value) for value in getattr(mesh, "grid_global_offset", (0.0,) * array.ndim))
    grid_position = _component_position(record_component, array.ndim)

    data, axes, block_origin, block_spacing, block_grid_position = _reduce_to_2d(
        array,
        axis_labels=axis_labels,
        origin=origin,
        spacing=spacing,
        grid_position=grid_position,
        requested_plane=requested_plane,
    )
    patch, level = _patch_and_level(mesh_name)
    metadata = {"openpmd_mesh": mesh_name, "grid_position": block_grid_position}
    refinement_extent = _refinement_extent(
        level=level,
        axes=axes,
        block_extent=_extent_from_block_values(data, block_origin, block_spacing),
        radii=refinement_radii,
        domain_bounds=domain_bounds,
    )
    if refinement_extent is not None:
        metadata["refinement_extent"] = refinement_extent
    return GridBlock(
        data=data,
        axes=axes,
        origin=block_origin,
        spacing=block_spacing,
        patch=patch,
        level=level,
        metadata=metadata,
    )


def _to_field_block(
    array: np.ndarray,
    mesh: Any,
    record_component: Any | None = None,
    *,
    mesh_name: str,
    refinement_radii: tuple[float, ...],
    domain_bounds: dict[str, tuple[float, float]],
) -> GridBlock:
    data = np.asarray(np.squeeze(array), dtype=float)
    if data.ndim != 3:
        raise ValueError(f"Expected 3D openPMD data for plane interpolation, got shape {array.shape}.")
    axes = tuple(str(axis) for axis in getattr(mesh, "axis_labels", ()))
    if not axes:
        axes = tuple(f"axis{i}" for i in range(data.ndim))
    if len(axes) != data.ndim:
        raise ValueError(f"openPMD axis labels {axes!r} do not match data shape {data.shape}.")
    spacing = tuple(float(value) for value in getattr(mesh, "grid_spacing", (1.0,) * data.ndim))
    origin = tuple(float(value) for value in getattr(mesh, "grid_global_offset", (0.0,) * data.ndim))
    grid_position = _component_position(record_component, data.ndim)
    patch, level = _patch_and_level(mesh_name)
    metadata = {"openpmd_mesh": mesh_name, "grid_position": grid_position}
    refinement_bounds = _refinement_bounds(
        level=level,
        axes=axes,
        shape=data.shape,
        origin=origin,
        spacing=spacing,
        radii=refinement_radii,
        domain_bounds=domain_bounds,
    )
    if refinement_bounds is not None:
        metadata["refinement_bounds"] = refinement_bounds
    return GridBlock(
        data=data,
        axes=axes,
        origin=origin,
        spacing=spacing,
        patch=patch,
        level=level,
        metadata=metadata,
    )


def _reduce_to_2d(
    array: np.ndarray,
    *,
    axis_labels: tuple[str, ...],
    origin: tuple[float, ...],
    spacing: tuple[float, ...],
    grid_position: tuple[float, ...],
    requested_plane: str | None,
) -> tuple[np.ndarray, tuple[str, ...], tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
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
            tuple(grid_position[i] for i in kept),
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
        tuple(grid_position[i] for i in kept),
    )


def _component_position(record_component: Any, ndim: int) -> tuple[float, ...]:
    try:
        position = tuple(float(value) for value in getattr(record_component, "position"))
    except Exception:
        return (0.0,) * ndim
    if len(position) != ndim:
        raise ValueError(f"openPMD component position {position!r} does not match data dimensions {ndim}.")
    return position


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


def _read_boxinbox_radii(path: Path) -> tuple[float, ...]:
    for parfile in _candidate_parfiles(path):
        text = parfile.read_text(encoding="utf-8")
        match = BOXINBOX_RADIUS_RE.search(text)
        if match is None:
            continue
        values = []
        for value in match.group("values").split(","):
            values.append(float(value.strip()))
        return tuple(values)
    return ()


def _candidate_parfiles(path: Path) -> tuple[Path, ...]:
    roots = [path if path.is_dir() else path.parent]
    roots.extend(roots[0].parents[:4])
    parfiles: list[Path] = []
    for root in roots:
        parfiles.extend(sorted(root.glob("*.par")))
    return tuple(parfiles)


def _extent_from_block_values(
    data: np.ndarray,
    origin: tuple[float, ...],
    spacing: tuple[float, ...],
) -> tuple[float, float, float, float]:
    y_count, x_count = data.shape
    y0, x0 = origin
    dy, dx = spacing
    return (x0, x0 + dx * x_count, y0, y0 + dy * y_count)


def _refinement_extent(
    *,
    level: int | None,
    axes: tuple[str, ...],
    block_extent: tuple[float, float, float, float],
    radii: tuple[float, ...],
    domain_bounds: dict[str, tuple[float, float]],
) -> tuple[float, float, float, float] | None:
    if level == 0 and domain_bounds and len(axes) == 2:
        return _axis_bounds_extent(axes, block_extent, domain_bounds)
    if level is None or level >= len(radii):
        return None
    radius = radii[level]
    if radius <= 0:
        return block_extent

    x0, x1, y0, y1 = block_extent
    axis_limits = {
        "x": (-radius, radius),
        "y": (-radius, radius),
        "z": (-radius, radius),
    }
    if len(axes) != 2:
        return None
    y_axis, x_axis = axes
    rx0, rx1 = axis_limits.get(x_axis, (x0, x1))
    ry0, ry1 = axis_limits.get(y_axis, (y0, y1))
    return max(x0, rx0), min(x1, rx1), max(y0, ry0), min(y1, ry1)


def _refinement_bounds(
    *,
    level: int | None,
    axes: tuple[str, ...],
    shape: tuple[int, ...],
    origin: tuple[float, ...],
    spacing: tuple[float, ...],
    radii: tuple[float, ...],
    domain_bounds: dict[str, tuple[float, float]],
) -> dict[str, tuple[float, float]] | None:
    if len(axes) != 3:
        return None
    block_bounds = {
        axis: (origin[index], origin[index] + spacing[index] * shape[index])
        for index, axis in enumerate(axes)
    }
    if level == 0 and domain_bounds:
        return {
            axis: _clip_bounds(block_bounds[axis], domain_bounds.get(axis, block_bounds[axis]))
            for axis in axes
        }
    if level is None or level >= len(radii):
        return None
    radius = radii[level]
    if radius <= 0:
        target_bounds = {axis: block_bounds[axis] for axis in axes}
    else:
        target_bounds = {axis: (-radius, radius) for axis in axes}
    return {
        axis: _clip_bounds(block_bounds[axis], target_bounds[axis])
        for axis in axes
    }


def _clip_bounds(
    block_bounds: tuple[float, float],
    target_bounds: tuple[float, float],
) -> tuple[float, float]:
    return max(block_bounds[0], target_bounds[0]), min(block_bounds[1], target_bounds[1])


def _axis_bounds_extent(
    axes: tuple[str, ...],
    block_extent: tuple[float, float, float, float],
    bounds: dict[str, tuple[float, float]],
) -> tuple[float, float, float, float]:
    x0, x1, y0, y1 = block_extent
    y_axis, x_axis = axes
    bx0, bx1 = bounds.get(x_axis, (x0, x1))
    by0, by1 = bounds.get(y_axis, (y0, y1))
    return max(x0, bx0), min(x1, bx1), max(y0, by0), min(y1, by1)


def _read_carpetx_domain_bounds(path: Path) -> dict[str, tuple[float, float]]:
    for parfile in _candidate_parfiles(path):
        text = parfile.read_text(encoding="utf-8")
        values: dict[str, dict[str, float]] = {}
        for match in CARPETX_BOUND_RE.finditer(text):
            axis = match.group("axis")
            side = match.group("side")
            values.setdefault(axis, {})[side] = float(match.group("value"))
        bounds = {
            axis: (sides["min"], sides["max"])
            for axis, sides in values.items()
            if "min" in sides and "max" in sides
        }
        if bounds:
            return bounds
    return {}
