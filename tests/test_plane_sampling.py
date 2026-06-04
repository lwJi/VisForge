from __future__ import annotations

import numpy as np
import pytest

from visforge.data.model import FieldData, FieldInfo, GridBlock, PlaneSpec
from visforge.data.plane import sample_field_on_plane


def test_sample_field_on_plane_linear_interpolates_3d_block() -> None:
    z, y, x = np.meshgrid(
        np.arange(3, dtype=float),
        np.arange(3, dtype=float),
        np.arange(3, dtype=float),
        indexing="ij",
    )
    block = GridBlock(
        data=x + 10.0 * y + 100.0 * z,
        axes=("z", "y", "x"),
        origin=(0.0, 0.0, 0.0),
        spacing=(1.0, 1.0, 1.0),
    )
    field = FieldData(
        field=FieldInfo(name="rho", dimensions=3),
        iteration=7,
        time=1.25,
        blocks=(block,),
    )
    plane = PlaneSpec(
        origin=(1.0, 1.0, 1.0),
        normal=(0.0, 0.0, 1.0),
        up=(0.0, 1.0, 0.0),
        size=(2.0, 2.0),
        resolution=(2, 2),
        interpolation="linear",
    )

    sampled = sample_field_on_plane(field, plane)

    assert sampled.field.name == "rho"
    assert sampled.iteration == 7
    assert sampled.time == 1.25
    assert sampled.plane == "sample_plane"
    np.testing.assert_allclose(
        sampled.blocks[0].data,
        np.array(
            [
                [105.5, 106.5],
                [115.5, 116.5],
            ]
        ),
    )
    assert sampled.blocks[0].axes == ("v", "u")
    assert sampled.blocks[0].origin == (-1.0, -1.0)
    assert sampled.blocks[0].spacing == (1.0, 1.0)


def test_sample_field_on_plane_nearest_uses_finer_blocks_last() -> None:
    coarse = GridBlock(
        data=np.zeros((3, 3, 3), dtype=float),
        axes=("z", "y", "x"),
        origin=(0.0, 0.0, 0.0),
        spacing=(1.0, 1.0, 1.0),
        level=0,
    )
    fine = GridBlock(
        data=np.full((2, 2, 2), 9.0, dtype=float),
        axes=("z", "y", "x"),
        origin=(0.5, 0.5, 0.5),
        spacing=(0.5, 0.5, 0.5),
        level=1,
    )
    field = FieldData(
        field=FieldInfo(name="rho", dimensions=3),
        iteration=0,
        time=None,
        blocks=(coarse, fine),
    )
    plane = PlaneSpec(
        origin=(0.75, 0.75, 0.75),
        normal=(0.0, 0.0, 1.0),
        up=(0.0, 1.0, 0.0),
        size=(0.5, 0.5),
        resolution=(1, 1),
        interpolation="nearest",
    )

    sampled = sample_field_on_plane(field, plane)

    assert sampled.blocks[0].data.tolist() == [[9.0]]


def test_sample_field_on_plane_respects_refinement_bounds() -> None:
    coarse = GridBlock(
        data=np.full((4, 4, 4), 1.0, dtype=float),
        axes=("z", "y", "x"),
        origin=(-1.5, -1.5, -1.5),
        spacing=(1.0, 1.0, 1.0),
        level=0,
    )
    fine = GridBlock(
        data=np.full((4, 4, 4), 9.0, dtype=float),
        axes=("z", "y", "x"),
        origin=(-1.5, -1.5, -1.5),
        spacing=(1.0, 1.0, 1.0),
        level=1,
        metadata={"refinement_bounds": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (-0.5, 0.5)}},
    )
    field = FieldData(
        field=FieldInfo(name="rho", dimensions=3),
        iteration=0,
        time=None,
        blocks=(coarse, fine),
    )
    plane = PlaneSpec(
        origin=(0.0, 0.0, 0.0),
        normal=(0.0, 0.0, 1.0),
        up=(0.0, 1.0, 0.0),
        size=(3.0, 3.0),
        resolution=(3, 3),
        interpolation="nearest",
    )

    sampled = sample_field_on_plane(field, plane)

    np.testing.assert_allclose(
        sampled.blocks[0].data,
        np.array(
            [
                [1.0, 1.0, 1.0],
                [1.0, 9.0, 1.0],
                [1.0, 1.0, 1.0],
            ]
        ),
    )


def test_sample_field_on_plane_rejects_parallel_up_vector() -> None:
    field = FieldData(
        field=FieldInfo(name="rho", dimensions=3),
        iteration=0,
        time=None,
        blocks=(),
    )
    plane = PlaneSpec(
        origin=(0.0, 0.0, 0.0),
        normal=(0.0, 0.0, 1.0),
        up=(0.0, 0.0, 2.0),
        size=(1.0, 1.0),
        resolution=(1, 1),
    )

    with pytest.raises(ValueError, match="sample_plane.up"):
        sample_field_on_plane(field, plane)
