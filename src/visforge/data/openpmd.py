"""openPMD backend for CarpetX `.bp5` output."""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
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


@dataclass(frozen=True)
class ChunkBox:
    slices: tuple[slice, ...]
    valid_slices: tuple[slice, ...]
    patch: int | None = None
    source_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class WrittenChunk:
    offset: tuple[int, ...]
    extent: tuple[int, ...]
    source_id: int | None = None


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
                blocks.extend(
                    _to_slice_blocks(
                        array,
                        mesh,
                        record_component,
                        mesh_name=mesh_name,
                        requested_plane=plane or file.plane,
                    )
                )

            if not blocks:
                raise ValueError(f"Field {field!r} was not found in {file.path}")

            annotated_blocks = _annotate_amr_coverage(tuple(blocks))
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
                blocks=annotated_blocks,
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
                blocks.extend(
                    _to_field_blocks(
                        array,
                        mesh,
                        record_component,
                        mesh_name=mesh_name,
                    )
                )

            if not blocks:
                raise ValueError(f"Field {field!r} was not found in {file.path}")

            annotated_blocks = _annotate_amr_coverage(tuple(blocks))
            return FieldData(
                field=FieldInfo(
                    name=selected_component or field,
                    components=((selected_component,) if selected_component is not None else ()),
                    dimensions=3,
                    metadata={"openpmd_record": field},
                ),
                iteration=file.iteration if file.iteration is not None else int(_first_iteration_key(series)),
                time=float(time) if time is not None else None,
                blocks=annotated_blocks,
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


def _to_slice_blocks(
    array: np.ndarray,
    mesh: Any,
    record_component: Any | None = None,
    *,
    mesh_name: str,
    requested_plane: str | None,
) -> tuple[GridBlock, ...]:
    axis_labels = tuple(str(axis) for axis in getattr(mesh, "axis_labels", ()))
    spacing = tuple(float(value) for value in getattr(mesh, "grid_spacing", (1.0,) * array.ndim))
    origin = tuple(float(value) for value in getattr(mesh, "grid_global_offset", (0.0,) * array.ndim))
    grid_position = _component_position(record_component, array.ndim)
    patch, level = _patch_and_level(mesh_name)
    blocks: list[GridBlock] = []
    for chunk_box in _chunk_boxes(record_component, array.shape):
        chunk_array, chunk_origin = _slice_array(array, origin, spacing, chunk_box.slices)
        valid_array, valid_origin = _slice_array(array, origin, spacing, chunk_box.valid_slices)
        data, axes, block_origin, block_spacing, block_grid_position, kept_dims = _reduce_to_2d(
            chunk_array,
            axis_labels=axis_labels,
            origin=chunk_origin,
            spacing=spacing,
            grid_position=grid_position,
            requested_plane=requested_plane,
        )
        valid_data, _, valid_block_origin, _, _, _ = _reduce_to_2d(
            valid_array,
            axis_labels=axis_labels,
            origin=valid_origin,
            spacing=spacing,
            grid_position=grid_position,
            requested_plane=requested_plane,
        )
        metadata = {
            "openpmd_mesh": mesh_name,
            "grid_position": block_grid_position,
            "amr_extent": _extent_from_block_values(valid_data, valid_block_origin, block_spacing),
            "amr_valid_slices": _local_valid_slices(
                chunk_box.slices,
                chunk_box.valid_slices,
                kept_dims=kept_dims,
            ),
        }
        if chunk_box.source_ids:
            metadata["openpmd_source_ids"] = chunk_box.source_ids
        blocks.append(
            GridBlock(
                data=data,
                axes=axes,
                origin=block_origin,
                spacing=block_spacing,
                patch=chunk_box.patch if chunk_box.patch is not None else patch,
                level=level,
                metadata=metadata,
            )
        )
    return tuple(blocks)


def _to_slice_block(
    array: np.ndarray,
    mesh: Any,
    record_component: Any | None = None,
    *,
    mesh_name: str,
    requested_plane: str | None,
) -> GridBlock:
    return _to_slice_blocks(
        array,
        mesh,
        record_component,
        mesh_name=mesh_name,
        requested_plane=requested_plane,
    )[0]


def _to_field_blocks(
    array: np.ndarray,
    mesh: Any,
    record_component: Any | None = None,
    *,
    mesh_name: str,
) -> tuple[GridBlock, ...]:
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
    blocks: list[GridBlock] = []
    for chunk_box in _chunk_boxes(record_component, data.shape):
        chunk_data, chunk_origin = _slice_array(data, origin, spacing, chunk_box.slices)
        valid_data, valid_origin = _slice_array(data, origin, spacing, chunk_box.valid_slices)
        metadata = {
            "openpmd_mesh": mesh_name,
            "grid_position": grid_position,
            "amr_bounds": _bounds_from_block_values(
                axes=axes,
                shape=valid_data.shape,
                origin=valid_origin,
                spacing=spacing,
            ),
            "amr_data_bounds": _bounds_from_block_values(
                axes=axes,
                shape=chunk_data.shape,
                origin=chunk_origin,
                spacing=spacing,
            ),
        }
        if chunk_box.source_ids:
            metadata["openpmd_source_ids"] = chunk_box.source_ids
        blocks.append(
            GridBlock(
                data=chunk_data,
                axes=axes,
                origin=chunk_origin,
                spacing=spacing,
                patch=chunk_box.patch if chunk_box.patch is not None else patch,
                level=level,
                metadata=metadata,
            )
        )
    return tuple(blocks)


def _to_field_block(
    array: np.ndarray,
    mesh: Any,
    record_component: Any | None = None,
    *,
    mesh_name: str,
) -> GridBlock:
    return _to_field_blocks(
        array,
        mesh,
        record_component,
        mesh_name=mesh_name,
    )[0]


def _chunk_boxes(record_component: Any | None, shape: tuple[int, ...]) -> tuple[ChunkBox, ...]:
    chunks = _available_chunks(record_component)
    if not chunks:
        slices = tuple(slice(0, size) for size in shape)
        return (ChunkBox(slices, valid_slices=slices),)

    components = _connected_chunk_components(chunks, shape)
    boxes = [
        _component_chunk_box(component, shape, patch=index)
        for index, component in enumerate(components)
    ]
    return tuple(box for box in boxes if _slice_volume(box.slices) > 0) or (
        ChunkBox(
            tuple(slice(0, size) for size in shape),
            valid_slices=tuple(slice(0, size) for size in shape),
        ),
    )


def _available_chunks(record_component: Any | None) -> tuple[WrittenChunk, ...]:
    if record_component is None:
        return ()
    try:
        chunks = tuple(record_component.available_chunks())
    except Exception:
        return ()
    written: list[WrittenChunk] = []
    for chunk in chunks:
        try:
            offset = tuple(int(value) for value in getattr(chunk, "offset"))
            extent = tuple(int(value) for value in getattr(chunk, "extent"))
        except Exception:
            continue
        written.append(WrittenChunk(offset=offset, extent=extent, source_id=_chunk_source_id(chunk)))
    return tuple(written)


def _chunk_source_id(chunk: Any) -> int | None:
    try:
        return int(getattr(chunk, "source_id"))
    except Exception:
        return None


def _connected_chunk_components(
    chunks: tuple[WrittenChunk, ...],
    shape: tuple[int, ...],
) -> tuple[tuple[WrittenChunk, ...], ...]:
    valid_chunks = tuple(
        chunk
        for chunk in chunks
        if len(chunk.offset) == len(shape)
        and len(chunk.extent) == len(shape)
        and all(value > 0 for value in chunk.extent)
    )
    remaining = set(range(len(valid_chunks)))
    components: list[tuple[WrittenChunk, ...]] = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        component_indices = [start]
        stack = [start]
        while stack:
            current = stack.pop()
            connected = [
                index
                for index in tuple(remaining)
                if _chunks_touch(valid_chunks[current], valid_chunks[index])
            ]
            for index in connected:
                remaining.remove(index)
                stack.append(index)
                component_indices.append(index)
        components.append(tuple(valid_chunks[index] for index in sorted(component_indices)))
    return tuple(components)


def _chunks_touch(left: WrittenChunk, right: WrittenChunk) -> bool:
    touching_dimensions = 0
    for dim in range(len(left.offset)):
        left_start = left.offset[dim]
        left_stop = left_start + left.extent[dim]
        right_start = right.offset[dim]
        right_stop = right_start + right.extent[dim]
        if left_stop < right_start or right_stop < left_start:
            return False
        if left_stop == right_start or right_stop == left_start:
            touching_dimensions += 1
    return touching_dimensions <= 1


def _chunks_overlap(left: WrittenChunk, right: WrittenChunk) -> bool:
    for dim in range(len(left.offset)):
        left_start = left.offset[dim]
        left_stop = left_start + left.extent[dim]
        right_start = right.offset[dim]
        right_stop = right_start + right.extent[dim]
        if left_stop <= right_start or right_stop <= left_start:
            return False
    return True


def _component_chunk_box(
    chunks: tuple[WrittenChunk, ...],
    shape: tuple[int, ...],
    *,
    patch: int,
) -> ChunkBox:
    valid_slices = _chunk_group_slices(chunks, shape)
    source_ids = tuple(sorted({chunk.source_id for chunk in chunks if chunk.source_id is not None}))
    return ChunkBox(
        slices=_expand_slices(valid_slices, shape, halo=1),
        valid_slices=valid_slices,
        patch=patch,
        source_ids=source_ids,
    )


def _chunk_group_slices(
    chunks: tuple[WrittenChunk, ...],
    shape: tuple[int, ...],
) -> tuple[slice, ...]:
    starts = [min(chunk.offset[dim] for chunk in chunks) for dim in range(len(shape))]
    stops = [
        max(chunk.offset[dim] + chunk.extent[dim] for chunk in chunks)
        for dim in range(len(shape))
    ]
    return tuple(
        slice(max(0, start), min(shape[dim], stop))
        for dim, (start, stop) in enumerate(zip(starts, stops))
    )


def _expand_slices(
    slices: tuple[slice, ...],
    shape: tuple[int, ...],
    *,
    halo: int,
) -> tuple[slice, ...]:
    return tuple(
        slice(
            max(0, int(item.start or 0) - halo),
            min(shape[dim], int(item.stop or 0) + halo),
        )
        for dim, item in enumerate(slices)
    )


def _slice_volume(slices: tuple[slice, ...]) -> int:
    volume = 1
    for item in slices:
        volume *= max(0, int(item.stop or 0) - int(item.start or 0))
    return volume


def _slice_array(
    array: np.ndarray,
    origin: tuple[float, ...],
    spacing: tuple[float, ...],
    slices: tuple[slice, ...],
) -> tuple[np.ndarray, tuple[float, ...]]:
    data = np.asarray(array[slices], dtype=float)
    sliced_origin = tuple(
        origin[dim] + spacing[dim] * float(item.start or 0)
        for dim, item in enumerate(slices)
    )
    return data, sliced_origin


def _reduce_to_2d(
    array: np.ndarray,
    *,
    axis_labels: tuple[str, ...],
    origin: tuple[float, ...],
    spacing: tuple[float, ...],
    grid_position: tuple[float, ...],
    requested_plane: str | None,
) -> tuple[
    np.ndarray,
    tuple[str, ...],
    tuple[float, ...],
    tuple[float, ...],
    tuple[float, ...],
    tuple[int, ...],
]:
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
            tuple(kept),
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
        tuple(kept),
    )


def _local_valid_slices(
    data_slices: tuple[slice, ...],
    valid_slices: tuple[slice, ...],
    *,
    kept_dims: tuple[int, ...],
) -> tuple[tuple[int, int], ...]:
    return tuple(
        (
            int(valid_slices[dim].start or 0) - int(data_slices[dim].start or 0),
            int(valid_slices[dim].stop or 0) - int(data_slices[dim].start or 0),
        )
        for dim in kept_dims
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


def _extent_from_block_values(
    data: np.ndarray,
    origin: tuple[float, ...],
    spacing: tuple[float, ...],
) -> tuple[float, float, float, float]:
    y_count, x_count = data.shape
    y0, x0 = origin
    dy, dx = spacing
    return (x0, x0 + dx * x_count, y0, y0 + dy * y_count)


def _bounds_from_block_values(
    *,
    axes: tuple[str, ...],
    shape: tuple[int, ...],
    origin: tuple[float, ...],
    spacing: tuple[float, ...],
) -> dict[str, tuple[float, float]]:
    return {
        axis: (origin[index], origin[index] + spacing[index] * shape[index])
        for index, axis in enumerate(axes)
    }


def _annotate_amr_coverage(blocks: tuple[GridBlock, ...]) -> tuple[GridBlock, ...]:
    annotated: list[GridBlock] = []
    for block in blocks:
        if block.level is None:
            annotated.append(block)
            continue
        if block.data.ndim == 2:
            covered_extents = _finer_covered_extents(block, blocks)
            if covered_extents:
                annotated.append(
                    replace(block, metadata={**block.metadata, "covered_extents": covered_extents})
                )
                continue
        if block.data.ndim == 3:
            covered_bounds = _finer_covered_bounds(block, blocks)
            if covered_bounds:
                annotated.append(
                    replace(block, metadata={**block.metadata, "covered_bounds": covered_bounds})
                )
                continue
        annotated.append(block)
    return tuple(annotated)


def _finer_covered_extents(
    block: GridBlock,
    blocks: tuple[GridBlock, ...],
) -> tuple[tuple[float, float, float, float], ...]:
    block_extent = _metadata_extent(block)
    if block_extent is None:
        return ()
    covered: list[tuple[float, float, float, float]] = []
    for candidate in blocks:
        if candidate is block or candidate.data.ndim != 2:
            continue
        if candidate.level is None or candidate.level <= block.level:
            continue
        if candidate.axes != block.axes:
            continue
        candidate_extent = _metadata_extent(candidate)
        if candidate_extent is None:
            continue
        overlap = _intersect_extent(block_extent, candidate_extent)
        if overlap is not None:
            covered.append(overlap)
    return tuple(covered)


def _metadata_extent(block: GridBlock) -> tuple[float, float, float, float] | None:
    extent = block.metadata.get("amr_extent") or block.metadata.get("refinement_extent")
    if extent is None:
        return None
    return tuple(float(value) for value in extent)


def _intersect_extent(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> tuple[float, float, float, float] | None:
    x0 = max(left[0], right[0])
    x1 = min(left[1], right[1])
    y0 = max(left[2], right[2])
    y1 = min(left[3], right[3])
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, x1, y0, y1


def _finer_covered_bounds(
    block: GridBlock,
    blocks: tuple[GridBlock, ...],
) -> tuple[dict[str, tuple[float, float]], ...]:
    block_bounds = _metadata_bounds(block)
    if not block_bounds:
        return ()
    covered: list[dict[str, tuple[float, float]]] = []
    for candidate in blocks:
        if candidate is block or candidate.data.ndim != 3:
            continue
        if candidate.level is None or candidate.level <= block.level:
            continue
        candidate_bounds = _metadata_bounds(candidate)
        overlap = _intersect_bounds(block_bounds, candidate_bounds)
        if overlap:
            covered.append(overlap)
    return tuple(covered)


def _metadata_bounds(block: GridBlock) -> dict[str, tuple[float, float]]:
    bounds = block.metadata.get("amr_bounds") or block.metadata.get("refinement_bounds")
    if not bounds:
        return {}
    return {
        str(axis): (float(axis_bounds[0]), float(axis_bounds[1]))
        for axis, axis_bounds in bounds.items()
    }


def _intersect_bounds(
    left: dict[str, tuple[float, float]],
    right: dict[str, tuple[float, float]],
) -> dict[str, tuple[float, float]]:
    if not left or not right or set(left) != set(right):
        return {}
    overlap: dict[str, tuple[float, float]] = {}
    for axis, left_bounds in left.items():
        right_bounds = right[axis]
        lower = max(left_bounds[0], right_bounds[0])
        upper = min(left_bounds[1], right_bounds[1])
        if upper <= lower:
            return {}
        overlap[axis] = (lower, upper)
    return overlap
