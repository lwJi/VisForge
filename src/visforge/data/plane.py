"""Interpolation from 3D field data onto user-defined 2D planes."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from visforge.data.model import FieldData, GridBlock, PlaneSpec, SliceData

AXIS_INDEX = {"x": 0, "y": 1, "z": 2}


def sample_field_on_plane(field_data: FieldData, plane: PlaneSpec) -> SliceData:
    """Sample a 3D scalar field onto a regular user-defined 2D plane."""

    origin, right, up = _plane_basis(plane)
    if not field_data.blocks:
        raise ValueError("Cannot sample a user-defined plane from FieldData with no blocks.")
    width, height = _positive_size(plane.size)
    nx, ny = _positive_resolution(plane.resolution)
    du = width / nx
    dv = height / ny

    u = -0.5 * width + du * (np.arange(nx, dtype=float) + 0.5)
    v = -0.5 * height + dv * (np.arange(ny, dtype=float) + 0.5)
    uu, vv = np.meshgrid(u, v, indexing="xy")
    points = origin + uu[..., np.newaxis] * right + vv[..., np.newaxis] * up
    flat_points = points.reshape(-1, 3)

    values = np.full(flat_points.shape[0], np.nan, dtype=float)
    for block in sorted(field_data.blocks, key=lambda item: (item.level or 0, item.patch or 0)):
        block_values, mask = _sample_block(block, flat_points, interpolation=plane.interpolation)
        values[mask] = block_values[mask]

    data = values.reshape(ny, nx)
    block = GridBlock(
        data=data,
        axes=("v", "u"),
        origin=(-0.5 * height, -0.5 * width),
        spacing=(dv, du),
        metadata={
            "sample_plane": {
                "origin": plane.origin,
                "normal": plane.normal,
                "up": plane.up,
                "size": plane.size,
                "resolution": plane.resolution,
                "interpolation": plane.interpolation,
            }
        },
    )
    return SliceData(
        field=field_data.field,
        iteration=field_data.iteration,
        time=field_data.time,
        plane="sample_plane",
        blocks=(block,),
        metadata={**field_data.metadata, "sample_plane": block.metadata["sample_plane"]},
    )


def _plane_basis(plane: PlaneSpec) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    origin = _vector3(plane.origin, "sample_plane.origin")
    normal = _unit_vector(_vector3(plane.normal, "sample_plane.normal"), "sample_plane.normal")
    up_input = _vector3(plane.up, "sample_plane.up")
    up_projected = up_input - np.dot(up_input, normal) * normal
    up = _unit_vector(up_projected, "sample_plane.up")
    right = np.cross(up, normal)
    return origin, right, up


def _vector3(values: tuple[float, float, float], name: str) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=float)
    if array.shape != (3,):
        raise ValueError(f"{name} must contain exactly three numeric values.")
    return array


def _unit_vector(vector: NDArray[np.float64], name: str) -> NDArray[np.float64]:
    norm = float(np.linalg.norm(vector))
    if np.isclose(norm, 0.0):
        raise ValueError(f"{name} must not be the zero vector.")
    return vector / norm


def _positive_size(size: tuple[float, float]) -> tuple[float, float]:
    width, height = float(size[0]), float(size[1])
    if width <= 0.0 or height <= 0.0:
        raise ValueError("sample_plane.size values must be positive.")
    return width, height


def _positive_resolution(resolution: tuple[int, int]) -> tuple[int, int]:
    nx, ny = int(resolution[0]), int(resolution[1])
    if nx <= 0 or ny <= 0:
        raise ValueError("sample_plane.resolution values must be positive.")
    return nx, ny


def _sample_block(
    block: GridBlock,
    points: NDArray[np.float64],
    *,
    interpolation: str,
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    if block.data.ndim != 3 or len(block.axes) != 3:
        raise ValueError("User-defined plane interpolation requires 3D GridBlock data.")
    axis_indices = _axis_indices(block.axes)
    coordinates = np.column_stack([points[:, axis_indices[axis]] for axis in block.axes])
    origin = np.asarray(block.origin, dtype=float)
    spacing = np.asarray(block.spacing, dtype=float)
    if np.any(spacing == 0.0):
        raise ValueError("Cannot interpolate a block with zero grid spacing.")
    indices = (coordinates - origin) / spacing
    if interpolation == "nearest":
        values, mask = _nearest(block.data, indices)
        return values, mask & _refinement_mask(block, points)
    if interpolation == "linear":
        values, mask = _linear(block.data, indices)
        return values, mask & _refinement_mask(block, points)
    raise ValueError("sample_plane.interpolation must be 'linear' or 'nearest'.")


def _axis_indices(axes: tuple[str, ...]) -> dict[str, int]:
    missing = [axis for axis in axes if axis not in AXIS_INDEX]
    if missing:
        raise ValueError(f"Cannot map block axes {axes!r} onto x/y/z coordinates.")
    return AXIS_INDEX


def _refinement_mask(block: GridBlock, points: NDArray[np.float64]) -> NDArray[np.bool_]:
    bounds = block.metadata.get("refinement_bounds")
    if not bounds:
        return np.ones(points.shape[0], dtype=bool)
    mask = np.ones(points.shape[0], dtype=bool)
    for axis, index in AXIS_INDEX.items():
        if axis not in bounds:
            continue
        lower, upper = bounds[axis]
        mask &= points[:, index] >= float(lower)
        mask &= points[:, index] <= float(upper)
    return mask


def _nearest(
    data: NDArray[np.floating[Any]],
    indices: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    rounded = np.rint(indices).astype(int)
    mask = np.ones(indices.shape[0], dtype=bool)
    for dim, size in enumerate(data.shape):
        mask &= rounded[:, dim] >= 0
        mask &= rounded[:, dim] < size
    values = np.full(indices.shape[0], np.nan, dtype=float)
    if np.any(mask):
        inside = rounded[mask]
        values[mask] = data[inside[:, 0], inside[:, 1], inside[:, 2]]
    return values, mask


def _linear(
    data: NDArray[np.floating[Any]],
    indices: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    mask = np.ones(indices.shape[0], dtype=bool)
    lower = np.zeros(indices.shape, dtype=int)
    upper = np.zeros(indices.shape, dtype=int)
    weights = np.zeros(indices.shape, dtype=float)
    for dim, size in enumerate(data.shape):
        mask &= indices[:, dim] >= 0.0
        mask &= indices[:, dim] <= float(size - 1)
        if size == 1:
            lower[:, dim] = 0
            upper[:, dim] = 0
            weights[:, dim] = 0.0
            continue
        dim_lower = np.floor(indices[:, dim]).astype(int)
        dim_lower = np.clip(dim_lower, 0, size - 2)
        lower[:, dim] = dim_lower
        upper[:, dim] = dim_lower + 1
        weights[:, dim] = indices[:, dim] - dim_lower

    values = np.full(indices.shape[0], np.nan, dtype=float)
    if not np.any(mask):
        return values, mask

    inside_lower = lower[mask]
    inside_upper = upper[mask]
    inside_weights = weights[mask]
    accum = np.zeros(inside_lower.shape[0], dtype=float)
    for d0 in (0, 1):
        i0 = inside_upper[:, 0] if d0 else inside_lower[:, 0]
        w0 = inside_weights[:, 0] if d0 else 1.0 - inside_weights[:, 0]
        for d1 in (0, 1):
            i1 = inside_upper[:, 1] if d1 else inside_lower[:, 1]
            w1 = inside_weights[:, 1] if d1 else 1.0 - inside_weights[:, 1]
            for d2 in (0, 1):
                i2 = inside_upper[:, 2] if d2 else inside_lower[:, 2]
                w2 = inside_weights[:, 2] if d2 else 1.0 - inside_weights[:, 2]
                accum += data[i0, i1, i2] * w0 * w1 * w2
    values[mask] = accum
    return values, mask
