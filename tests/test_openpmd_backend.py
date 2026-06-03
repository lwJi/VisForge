from __future__ import annotations

import importlib.util

import pytest

from visforge.data.openpmd import OpenPMDBackend

pytestmark = pytest.mark.integration


def test_openpmd_backend_reads_native_plane(example_data_root) -> None:
    if importlib.util.find_spec("openpmd_api") is None:
        pytest.skip("openPMD-api is not installed")

    backend = OpenPMDBackend(example_data_root / "testoutput2d")
    assert backend.list_iterations() == (0,)
    assert {"gfc", "gfv"}.issubset({field.name for field in backend.list_fields(0)})

    data = backend.read_slice("gfc", iteration=0, plane="xy")
    assert data.plane == "xy"
    assert data.blocks
    assert data.blocks[0].data.ndim == 2
