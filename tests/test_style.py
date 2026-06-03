from __future__ import annotations

from visforge.plotting.style import (
    PLOT_FONT_FAMILY,
    axis_label,
    configure_matplotlib_style,
    field_label,
    plane_label,
)


def test_latex_style_labels() -> None:
    assert axis_label("x") == r"$x$"
    assert axis_label("radius_grid") == r"$\mathrm{radius\_grid}$"
    assert field_label("gfc") == r"$\mathrm{gfc}$"
    assert field_label("rho_0", "code_units") == r"$\mathrm{rho\_0}\,[\mathrm{code\_units}]$"
    assert plane_label("xz") == r"$xz$"


def test_pretty_font_style() -> None:
    configure_matplotlib_style()
    import matplotlib as mpl

    assert mpl.rcParams["font.family"] == ["serif"]
    assert mpl.rcParams["font.serif"][0] == PLOT_FONT_FAMILY
    assert mpl.rcParams["mathtext.fontset"] == "stix"
