from __future__ import annotations

from pathlib import Path

import pytest

from visforge.data.openpmd import (
    _carpetx_metadata_from_iteration,
    _component_name,
    _parse_boxinbox_radii,
    _parse_carpetx_domain_bounds,
)


class FakeMesh:
    def __iter__(self):
        return iter(("testsubcyclingmc2_rho", "testsubcyclingmc2_u"))


class FakeIteration:
    def __init__(self, attributes):
        self.attributes = attributes

    def get_attribute(self, name):
        if name not in self.attributes:
            raise RuntimeError(name)
        return self.attributes[name]


def test_component_name_requires_component_for_multi_component_record() -> None:
    with pytest.raises(ValueError, match="multiple components"):
        _component_name(FakeMesh(), "state", None)


def test_component_name_selects_requested_component() -> None:
    assert _component_name(FakeMesh(), "state", "testsubcyclingmc2_rho") == "testsubcyclingmc2_rho"
    assert _component_name(FakeMesh(), "state", "u") == "testsubcyclingmc2_u"


def test_parse_carpetx_metadata_from_all_parameters() -> None:
    parameters = """
CarpetX::xmin = -12
CarpetX::xmax = 12
CarpetX::ymin = -13.5
CarpetX::ymax = 13.5
CarpetX::zmin = -14
CarpetX::zmax = 14
BoxInBox::radius_1[0] = -1
BoxInBox::radius_1[1] = 3
BoxInBox::radius_1[2] = 1.5
BoxInBox::radius_1[3] = -1
"""

    assert _parse_carpetx_domain_bounds(parameters) == {
        "x": (-12.0, 12.0),
        "y": (-13.5, 13.5),
        "z": (-14.0, 14.0),
    }
    assert _parse_boxinbox_radii(parameters) == (-1.0, 3.0, 1.5, -1.0)


def test_carpetx_metadata_requires_all_parameters() -> None:
    with pytest.raises(ValueError, match="AllParameters"):
        _carpetx_metadata_from_iteration(FakeIteration({}), Path("missing.bp5"))
