from __future__ import annotations

import importlib.util

import pytest

from visforge.data.openpmd import OpenPMDBackend

pytestmark = pytest.mark.integration


def _skip_without_all_parameters(backend: OpenPMDBackend, *, iteration: int, plane: str) -> None:
    import openpmd_api as io

    file = backend._select_one_file(iteration=iteration, plane=plane)
    series = io.Series(str(file.path), io.Access.read_only)
    try:
        opmd_iteration = series.iterations[list(series.iterations)[0]]
        if "AllParameters" not in list(opmd_iteration.attributes):
            pytest.skip(f"{file.path} does not contain AllParameters")
    finally:
        series.close()


def test_openpmd_backend_reads_native_plane(example_data_root) -> None:
    if importlib.util.find_spec("openpmd_api") is None:
        pytest.skip("openPMD-api is not installed")

    backend = OpenPMDBackend(example_data_root / "testoutput2d")
    assert backend.list_iterations() == (0,)
    assert {"gfc", "gfv"}.issubset({field.name for field in backend.list_fields(0)})

    _skip_without_all_parameters(backend, iteration=0, plane="xy")
    data = backend.read_slice("gfc", iteration=0, plane="xy")
    assert data.plane == "xy"
    assert data.blocks
    assert data.blocks[0].data.ndim == 2


def test_openpmd_backend_reads_boxinbox_refinement_extents(example_data_root) -> None:
    if importlib.util.find_spec("openpmd_api") is None:
        pytest.skip("openPMD-api is not installed")

    backend = OpenPMDBackend(example_data_root / "testoutput2d")
    _skip_without_all_parameters(backend, iteration=0, plane="xz")
    data = backend.read_slice("gfc", iteration=0, plane="xz")
    extents = {
        block.level: block.metadata.get("refinement_extent")
        for block in data.blocks
    }
    assert extents[0] == (-12.0, 12.0, -12.0, 12.0)
    assert extents[1] == (-3.0, 3.0, -3.0, 3.0)
    assert extents[2] == (-1.5, 1.5, -1.5, 1.5)
