from __future__ import annotations

import numpy as np
import pytest

from visforge.data.openpmd import (
    _annotate_amr_coverage,
    _component_name,
    _to_field_block,
    _to_field_blocks,
    _to_slice_block,
    _to_slice_blocks,
)
from visforge.data.model import GridBlock


class FakeMesh:
    axis_labels = ("z", "y", "x")
    grid_spacing = (0.5, 1.0, 2.0)
    grid_global_offset = (-1.0, -2.0, -3.0)

    def __iter__(self):
        return iter(("testsubcyclingmc2_rho", "testsubcyclingmc2_u"))


class FakeRecordComponent:
    position = (0.5, 0.5, 0.5)


class FakeChunk:
    def __init__(self, offset, extent, source_id):
        self.offset = offset
        self.extent = extent
        self.source_id = source_id


class FakeChunkedRecordComponent:
    position = (0.5, 0.5, 0.5)

    def __init__(self, chunks):
        self._chunks = tuple(chunks)

    def available_chunks(self):
        return self._chunks


class FakeSliceMesh:
    axis_labels = ("z", "y", "x")
    grid_spacing = (0.5, 1.0, 2.0)
    grid_global_offset = (-1.0, -2.0, -3.0)


def test_component_name_requires_component_for_multi_component_record() -> None:
    with pytest.raises(ValueError, match="multiple components"):
        _component_name(FakeMesh(), "state", None)


def test_component_name_selects_requested_component() -> None:
    assert _component_name(FakeMesh(), "state", "testsubcyclingmc2_rho") == "testsubcyclingmc2_rho"
    assert _component_name(FakeMesh(), "state", "u") == "testsubcyclingmc2_u"


def test_to_slice_block_records_amr_extent_from_mesh_geometry() -> None:
    block = _to_slice_block(
        np.zeros((1, 4, 5), dtype=float),
        FakeSliceMesh(),
        FakeRecordComponent(),
        mesh_name="rho_patch3_lev2",
        requested_plane="xy",
    )

    assert block.patch == 3
    assert block.level == 2
    assert block.axes == ("y", "x")
    assert block.metadata["amr_extent"] == (-3.0, 7.0, -2.0, 2.0)


def test_to_field_block_records_amr_bounds_from_mesh_geometry() -> None:
    block = _to_field_block(
        np.zeros((4, 5, 6), dtype=float),
        FakeMesh(),
        FakeRecordComponent(),
        mesh_name="rho_patch1_lev2",
    )

    assert block.metadata["amr_bounds"] == {
        "z": (-1.0, 1.0),
        "y": (-2.0, 3.0),
        "x": (-3.0, 9.0),
    }


def test_to_slice_blocks_split_chunk_source_boxes() -> None:
    array = np.arange(36, dtype=float).reshape(1, 6, 6)
    record_component = FakeChunkedRecordComponent(
        (
            FakeChunk(offset=(0, 1, 1), extent=(1, 2, 2), source_id=0),
            FakeChunk(offset=(0, 1, 3), extent=(1, 2, 1), source_id=0),
            FakeChunk(offset=(0, 4, 4), extent=(1, 1, 2), source_id=1),
        )
    )

    blocks = _to_slice_blocks(
        array,
        FakeSliceMesh(),
        record_component,
        mesh_name="rho_patch3_lev2",
        requested_plane="xy",
    )

    assert len(blocks) == 2
    assert blocks[0].patch == 0
    assert blocks[0].origin == (-2.0, -3.0)
    assert blocks[0].data.tolist() == [
        [0.0, 1.0, 2.0, 3.0, 4.0],
        [6.0, 7.0, 8.0, 9.0, 10.0],
        [12.0, 13.0, 14.0, 15.0, 16.0],
        [18.0, 19.0, 20.0, 21.0, 22.0],
    ]
    assert blocks[0].metadata["amr_extent"] == (-1.0, 5.0, -1.0, 1.0)
    assert blocks[0].metadata["amr_valid_slices"] == ((1, 3), (1, 4))
    assert blocks[1].patch == 1
    assert blocks[1].origin == (1.0, 3.0)
    assert blocks[1].data.tolist() == [
        [21.0, 22.0, 23.0],
        [27.0, 28.0, 29.0],
        [33.0, 34.0, 35.0],
    ]
    assert blocks[1].metadata["amr_extent"] == (5.0, 9.0, 2.0, 3.0)
    assert blocks[1].metadata["amr_valid_slices"] == ((1, 2), (1, 3))


def test_to_field_blocks_split_chunk_source_boxes() -> None:
    array = np.arange(4 * 5 * 6, dtype=float).reshape(4, 5, 6)
    record_component = FakeChunkedRecordComponent(
        (
            FakeChunk(offset=(1, 2, 3), extent=(1, 2, 2), source_id=7),
            FakeChunk(offset=(2, 2, 3), extent=(1, 2, 2), source_id=7),
        )
    )

    blocks = _to_field_blocks(
        array,
        FakeMesh(),
        record_component,
        mesh_name="rho_patch1_lev2",
    )

    assert len(blocks) == 1
    assert blocks[0].patch == 0
    assert blocks[0].origin == (-1.0, -1.0, 1.0)
    assert blocks[0].data.shape == (4, 4, 4)
    assert blocks[0].metadata["amr_bounds"] == {
        "z": (-0.5, 0.5),
        "y": (0.0, 2.0),
        "x": (3.0, 7.0),
    }
    assert blocks[0].metadata["amr_data_bounds"] == {
        "z": (-1.0, 1.0),
        "y": (-1.0, 3.0),
        "x": (1.0, 9.0),
    }
    assert blocks[0].metadata["openpmd_source_ids"] == (7,)


def test_to_field_blocks_split_disconnected_chunks_with_same_source_id() -> None:
    array = np.arange(5 * 5 * 5, dtype=float).reshape(5, 5, 5)
    record_component = FakeChunkedRecordComponent(
        (
            FakeChunk(offset=(0, 0, 0), extent=(1, 1, 1), source_id=2),
            FakeChunk(offset=(4, 4, 4), extent=(1, 1, 1), source_id=2),
        )
    )

    blocks = _to_field_blocks(
        array,
        FakeMesh(),
        record_component,
        mesh_name="rho_patch1_lev2",
    )

    assert len(blocks) == 2
    assert blocks[0].metadata["amr_bounds"] == {
        "z": (-1.0, -0.5),
        "y": (-2.0, -1.0),
        "x": (-3.0, -1.0),
    }
    assert blocks[1].metadata["amr_bounds"] == {
        "z": (1.0, 1.5),
        "y": (2.0, 3.0),
        "x": (5.0, 7.0),
    }


def test_annotate_amr_coverage_supports_multiple_boxes_per_level() -> None:
    coarse = GridBlock(
        data=np.zeros((4, 4), dtype=float),
        axes=("y", "x"),
        origin=(0.0, 0.0),
        spacing=(1.0, 1.0),
        level=0,
        metadata={"amr_extent": (0.0, 4.0, 0.0, 4.0)},
    )
    fine_left = GridBlock(
        data=np.zeros((2, 2), dtype=float),
        axes=("y", "x"),
        origin=(0.0, 0.0),
        spacing=(0.5, 0.5),
        level=1,
        metadata={"amr_extent": (0.0, 1.0, 0.0, 1.0)},
    )
    fine_right = GridBlock(
        data=np.zeros((2, 2), dtype=float),
        axes=("y", "x"),
        origin=(2.0, 2.0),
        spacing=(0.5, 0.5),
        level=1,
        metadata={"amr_extent": (2.0, 3.0, 2.0, 3.0)},
    )

    annotated = _annotate_amr_coverage((coarse, fine_left, fine_right))

    assert annotated[0].metadata["covered_extents"] == (
        (0.0, 1.0, 0.0, 1.0),
        (2.0, 3.0, 2.0, 3.0),
    )
