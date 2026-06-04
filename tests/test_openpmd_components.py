from __future__ import annotations

import pytest

from visforge.data.openpmd import _component_name


class FakeMesh:
    def __iter__(self):
        return iter(("testsubcyclingmc2_rho", "testsubcyclingmc2_u"))


def test_component_name_requires_component_for_multi_component_record() -> None:
    with pytest.raises(ValueError, match="multiple components"):
        _component_name(FakeMesh(), "state", None)


def test_component_name_selects_requested_component() -> None:
    assert _component_name(FakeMesh(), "state", "testsubcyclingmc2_rho") == "testsubcyclingmc2_rho"
    assert _component_name(FakeMesh(), "state", "u") == "testsubcyclingmc2_u"
