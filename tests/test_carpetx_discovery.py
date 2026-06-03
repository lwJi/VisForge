from __future__ import annotations

import pytest

from visforge.data.carpetx import discover

pytestmark = pytest.mark.integration


def test_discover_example_data(example_data_root) -> None:
    index = discover(example_data_root / "testoutput2d")
    assert "openpmd" in index.backends
    assert "silo" in index.backends
    assert "tsv" in index.backends
    assert 0 in index.iterations
    assert {"x", "y", "z"}.issubset(index.axes)
    assert {"xy", "xz", "yz"}.issubset(index.planes)
