from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from visforge.data.model import FieldInfo, GridBlock, SliceData
from visforge.plotting.base import PlotLabels
from visforge.plotting.scalar import (
    _color_limits,
    _display_data,
    _mesh_line_positions,
    _valid_data_and_extent,
    plot_scalar_slice,
)


def test_plot_scalar_slice_writes_png(tmp_path: Path) -> None:
    block = GridBlock(
        data=np.arange(9, dtype=float).reshape(3, 3),
        axes=("y", "x"),
        origin=(0.0, 0.0),
        spacing=(0.5, 0.5),
    )
    slice_data = SliceData(
        field=FieldInfo(name="rho"),
        iteration=0,
        time=0.0,
        plane="xy",
        blocks=(block,),
    )
    output = tmp_path / "slice.png"
    result = plot_scalar_slice(slice_data, output=output)
    assert result.output == output.resolve()
    assert output.stat().st_size > 0
    assert result.axes.get_xlabel() == r"$x$"
    assert result.axes.get_ylabel() == r"$y$"
    assert result.axes.get_aspect() == 1.0


def test_plot_scalar_slice_writes_custom_dpi(tmp_path: Path) -> None:
    block = GridBlock(
        data=np.arange(9, dtype=float).reshape(3, 3),
        axes=("y", "x"),
        origin=(0.0, 0.0),
        spacing=(0.5, 0.5),
    )
    slice_data = SliceData(
        field=FieldInfo(name="rho"),
        iteration=0,
        time=0.0,
        plane="xy",
        blocks=(block,),
    )
    output = tmp_path / "slice_dpi.png"
    plot_scalar_slice(slice_data, output=output, dpi=240)

    with Image.open(output) as image:
        assert image.info["dpi"] == pytest.approx((240, 240), abs=1)


def test_plot_scalar_slice_applies_axis_ranges(tmp_path: Path) -> None:
    block = GridBlock(
        data=np.arange(9, dtype=float).reshape(3, 3),
        axes=("y", "x"),
        origin=(0.0, 0.0),
        spacing=(0.5, 0.5),
    )
    slice_data = SliceData(
        field=FieldInfo(name="rho"),
        iteration=0,
        time=0.0,
        plane="xy",
        blocks=(block,),
    )
    result = plot_scalar_slice(
        slice_data,
        output=tmp_path / "slice_zoom.png",
        xlim=(-0.25, 0.75),
        ylim=(0.25, 1.25),
    )
    assert result.axes.get_xlim() == (-0.25, 0.75)
    assert result.axes.get_ylim() == (0.25, 1.25)


def test_plot_scalar_slice_applies_label_overrides(tmp_path: Path) -> None:
    block = GridBlock(
        data=np.arange(9, dtype=float).reshape(3, 3),
        axes=("y", "x"),
        origin=(0.0, 0.0),
        spacing=(0.5, 0.5),
    )
    slice_data = SliceData(
        field=FieldInfo(name="rho"),
        iteration=0,
        time=0.0,
        plane="xy",
        blocks=(block,),
    )

    result = plot_scalar_slice(
        slice_data,
        output=tmp_path / "slice_labels.png",
        labels=PlotLabels(
            title="Custom slice",
            xlabel="Radius",
            ylabel="Height",
            colorbar="Density",
        ),
    )

    assert result.axes.get_title() == "Custom slice"
    assert result.axes.get_xlabel() == "Radius"
    assert result.axes.get_ylabel() == "Height"
    assert result.figure.axes[-1].get_ylabel() == "Density"


def test_plot_scalar_slice_formats_title_template(tmp_path: Path) -> None:
    block = GridBlock(
        data=np.arange(9, dtype=float).reshape(3, 3),
        axes=("y", "x"),
        origin=(0.0, 0.0),
        spacing=(0.5, 0.5),
    )
    slice_data = SliceData(
        field=FieldInfo(name="B_x"),
        iteration=7,
        time=1.25,
        plane="xz",
        blocks=(block,),
    )

    result = plot_scalar_slice(
        slice_data,
        output=tmp_path / "slice_title_template.png",
        labels=PlotLabels(title="$B_x$\niteration {iteration}, $t={time:g}$ on {plane}"),
    )

    assert result.axes.get_title() == "$B_x$\niteration 7, $t=1.25$ on xz"


def test_plot_scalar_slice_title_template_preserves_latex_braces(tmp_path: Path) -> None:
    block = GridBlock(
        data=np.arange(9, dtype=float).reshape(3, 3),
        axes=("y", "x"),
        origin=(0.0, 0.0),
        spacing=(0.5, 0.5),
    )
    slice_data = SliceData(
        field=FieldInfo(name="rho"),
        iteration=3,
        time=None,
        plane="xy",
        blocks=(block,),
    )

    result = plot_scalar_slice(
        slice_data,
        output=tmp_path / "slice_latex_title.png",
        labels=PlotLabels(title=r"$\rho_{\rm b}$, iteration {iteration}"),
    )

    assert result.axes.get_title() == r"$\rho_{\rm b}$, iteration 3"


def test_plot_scalar_slice_defaults_axis_ranges_to_coarsest_level(tmp_path: Path) -> None:
    coarse_left = GridBlock(
        data=np.ones((2, 2), dtype=float),
        axes=("y", "x"),
        origin=(0.0, 0.0),
        spacing=(1.0, 1.0),
        level=0,
        patch=0,
    )
    coarse_right = GridBlock(
        data=np.ones((2, 2), dtype=float),
        axes=("y", "x"),
        origin=(-1.0, 2.0),
        spacing=(1.0, 1.0),
        level=0,
        patch=1,
    )
    fine_outside_coarse_range = GridBlock(
        data=np.ones((2, 2), dtype=float),
        axes=("y", "x"),
        origin=(-10.0, -10.0),
        spacing=(1.0, 1.0),
        level=1,
        patch=0,
    )
    slice_data = SliceData(
        field=FieldInfo(name="rho"),
        iteration=0,
        time=0.0,
        plane="xy",
        blocks=(coarse_left, coarse_right, fine_outside_coarse_range),
    )

    result = plot_scalar_slice(slice_data, output=tmp_path / "slice_coarse_range.png")

    assert result.axes.get_xlim() == (0.0, 4.0)
    assert result.axes.get_ylim() == (-1.0, 2.0)


def test_plot_scalar_slice_uses_shared_norm_for_constant_blocks(tmp_path: Path) -> None:
    coarse = GridBlock(
        data=np.zeros((3, 3), dtype=float),
        axes=("y", "x"),
        origin=(-1.0, -1.0),
        spacing=(1.0, 1.0),
        level=0,
    )
    fine = GridBlock(
        data=np.zeros((3, 3), dtype=float),
        axes=("y", "x"),
        origin=(-0.5, -0.5),
        spacing=(0.5, 0.5),
        level=1,
    )
    slice_data = SliceData(
        field=FieldInfo(name="rho"),
        iteration=0,
        time=0.0,
        plane="xy",
        blocks=(coarse, fine),
    )

    result = plot_scalar_slice(slice_data, output=tmp_path / "constant.png")

    assert len(result.axes.images) == 2
    assert result.axes.images[0].norm is result.axes.images[1].norm
    assert result.axes.images[0].norm.vmin == -0.1
    assert result.axes.images[0].norm.vmax == 0.1


def test_plot_scalar_slice_uses_log_norm(tmp_path: Path) -> None:
    from matplotlib.colors import LogNorm

    block = GridBlock(
        data=np.array([[0.0, 1.0, 10.0], [100.0, np.nan, -5.0]], dtype=float),
        axes=("y", "x"),
        origin=(0.0, 0.0),
        spacing=(1.0, 1.0),
    )
    slice_data = SliceData(
        field=FieldInfo(name="rho"),
        iteration=0,
        time=0.0,
        plane="xy",
        blocks=(block,),
    )

    result = plot_scalar_slice(slice_data, output=tmp_path / "log.png", scale="log")

    assert isinstance(result.axes.images[0].norm, LogNorm)
    assert result.axes.images[0].norm.vmin > 0.0
    assert result.axes.images[0].norm.vmax > result.axes.images[0].norm.vmin


def test_plot_scalar_slice_uses_requested_colormap(tmp_path: Path) -> None:
    block = GridBlock(
        data=np.arange(9, dtype=float).reshape(3, 3),
        axes=("y", "x"),
        origin=(0.0, 0.0),
        spacing=(1.0, 1.0),
    )
    slice_data = SliceData(
        field=FieldInfo(name="rho"),
        iteration=0,
        time=0.0,
        plane="xy",
        blocks=(block,),
    )

    result = plot_scalar_slice(slice_data, output=tmp_path / "plasma.png", cmap="plasma")

    assert result.axes.images[0].get_cmap().name == "plasma"


def test_plot_scalar_slice_rejects_unknown_colormap(tmp_path: Path) -> None:
    block = GridBlock(
        data=np.arange(4, dtype=float).reshape(2, 2),
        axes=("y", "x"),
        origin=(0.0, 0.0),
        spacing=(1.0, 1.0),
    )
    slice_data = SliceData(
        field=FieldInfo(name="rho"),
        iteration=0,
        time=0.0,
        plane="xy",
        blocks=(block,),
    )

    with pytest.raises(ValueError, match="Unknown colormap 'not_a_cmap'"):
        plot_scalar_slice(slice_data, output=tmp_path / "bad.png", cmap="not_a_cmap")


def test_plot_scalar_slice_can_overlay_mesh(tmp_path: Path) -> None:
    block = GridBlock(
        data=np.arange(4, dtype=float).reshape(2, 2),
        axes=("y", "x"),
        origin=(-1.0, 2.0),
        spacing=(0.5, 0.25),
    )
    slice_data = SliceData(
        field=FieldInfo(name="rho"),
        iteration=0,
        time=0.0,
        plane="xy",
        blocks=(block,),
    )
    output = tmp_path / "slice_mesh.png"
    result = plot_scalar_slice(
        slice_data,
        output=output,
        show_mesh=True,
        mesh_linewidth=1.25,
        mesh_alpha=0.4,
    )
    assert result.output == output.resolve()
    assert output.stat().st_size > 0
    assert len(result.axes.patches) == 1
    assert len(result.axes.collections) == 2
    assert result.axes.patches[0].get_linewidth() == 1.25
    assert result.axes.collections[0].get_linewidths().tolist() == [1.25]
    assert result.axes.patches[0].get_alpha() == 0.4
    assert result.axes.collections[0].get_alpha() == 0.4
    assert result.axes.patches[0].get_path_effects() == []
    assert result.axes.collections[0].get_path_effects() in (None, [])


def test_mesh_line_positions_use_real_cell_spacing() -> None:
    block = GridBlock(
        data=np.zeros((2, 3), dtype=float),
        axes=("z", "x"),
        origin=(-1.0, 2.0),
        spacing=(0.5, 0.25),
    )
    xs, ys = _mesh_line_positions(block, extent=(2.0, 2.75, -1.0, 0.0), max_lines=None)
    assert xs.tolist() == [2.0, 2.25, 2.5, 2.75]
    assert ys.tolist() == [-1.0, -0.5, 0.0]


def test_valid_data_and_extent_crops_to_refinement_region() -> None:
    block = GridBlock(
        data=np.arange(25, dtype=float).reshape(5, 5),
        axes=("z", "x"),
        origin=(-2.0, -2.0),
        spacing=(1.0, 1.0),
        metadata={"refinement_extent": (-1.0, 2.0, -1.0, 2.0)},
    )
    data, extent = _valid_data_and_extent(block)
    assert data.shape == (3, 3)
    assert data.tolist() == [
        [6.0, 7.0, 8.0],
        [11.0, 12.0, 13.0],
        [16.0, 17.0, 18.0],
    ]
    assert extent == (-1.0, 2.0, -1.0, 2.0)


def test_valid_data_and_extent_masks_regions_covered_by_finer_boxes() -> None:
    block = GridBlock(
        data=np.arange(16, dtype=float).reshape(4, 4),
        axes=("y", "x"),
        origin=(0.0, 0.0),
        spacing=(1.0, 1.0),
        metadata={
            "amr_extent": (0.0, 4.0, 0.0, 4.0),
            "covered_extents": (
                (0.0, 1.0, 0.0, 1.0),
                (2.0, 3.0, 2.0, 3.0),
            ),
        },
    )

    data, extent = _valid_data_and_extent(block)

    assert extent == (0.0, 4.0, 0.0, 4.0)
    assert np.isnan(data[0, 0])
    assert np.isnan(data[2, 2])
    assert data[1, 2] == 6.0


def test_valid_data_and_extent_can_keep_covered_regions_for_rendering() -> None:
    block = GridBlock(
        data=np.arange(16, dtype=float).reshape(4, 4),
        axes=("y", "x"),
        origin=(0.0, 0.0),
        spacing=(1.0, 1.0),
        metadata={
            "amr_extent": (0.0, 4.0, 0.0, 4.0),
            "covered_extents": ((0.0, 1.0, 0.0, 1.0),),
        },
    )

    data, extent = _valid_data_and_extent(block, mask_covered=False)

    assert extent == (0.0, 4.0, 0.0, 4.0)
    assert data[0, 0] == 0.0
    assert not np.isnan(data).any()


def test_valid_data_and_extent_uses_explicit_valid_slices_before_extent_math() -> None:
    values = np.full((5, 5), 999.0, dtype=float)
    values[1:4, 1:4] = np.arange(9, dtype=float).reshape(3, 3)
    block = GridBlock(
        data=values,
        axes=("y", "x"),
        origin=(0.0, 0.0),
        spacing=(1.0, 1.0),
        metadata={
            "grid_position": (0.0, 0.0),
            "amr_extent": (1.0, 4.0, 1.0, 4.0),
            "amr_valid_slices": ((1, 4), (1, 4)),
        },
    )

    data, extent = _valid_data_and_extent(block, mask_covered=False)

    assert extent == (1.0, 4.0, 1.0, 4.0)
    assert data.tolist() == [
        [0.0, 1.0, 2.0],
        [3.0, 4.0, 5.0],
        [6.0, 7.0, 8.0],
    ]


def test_color_limits_expand_constant_data() -> None:
    block = GridBlock(
        data=np.full((2, 2), 5.0, dtype=float),
        axes=("z", "x"),
        origin=(0.0, 0.0),
        spacing=(1.0, 1.0),
    )

    assert _color_limits((block,), vmin=None, vmax=None) == (4.95, 5.05)


def test_color_limits_log_scale_uses_positive_data_only() -> None:
    block = GridBlock(
        data=np.array([[-10.0, 0.0, 1.0], [10.0, 100.0, np.inf]], dtype=float),
        axes=("z", "x"),
        origin=(0.0, 0.0),
        spacing=(1.0, 1.0),
    )

    lower, upper = _color_limits((block,), vmin=None, vmax=None, scale="log")

    assert lower == pytest.approx(1.18)
    assert upper == pytest.approx(98.2)


def test_color_limits_log_scale_rejects_non_positive_vmin() -> None:
    block = GridBlock(
        data=np.array([[1.0, 10.0]], dtype=float),
        axes=("z", "x"),
        origin=(0.0, 0.0),
        spacing=(1.0, 1.0),
    )

    with pytest.raises(ValueError, match="vmin"):
        _color_limits((block,), vmin=0.0, vmax=None, scale="log")


def test_display_data_masks_invalid_values() -> None:
    display = _display_data(
        np.array(
            [
                [1.0, np.nan],
                [np.inf, 2.0e100],
            ],
            dtype=float,
        )
    )

    assert np.ma.isMaskedArray(display)
    assert display.mask.tolist() == [
        [False, True],
        [True, True],
    ]
    assert display[0, 0] == 1.0


def test_display_data_masks_non_positive_log_values() -> None:
    display = _display_data(np.array([[0.0, -1.0, 2.0]], dtype=float), scale="log")

    assert np.ma.isMaskedArray(display)
    assert display.mask.tolist() == [[True, True, False]]


def test_valid_data_and_extent_respects_cell_centered_position() -> None:
    block = GridBlock(
        data=np.arange(81, dtype=float).reshape(9, 9),
        axes=("z", "x"),
        origin=(-2.0, -2.0),
        spacing=(0.5, 0.5),
        metadata={
            "grid_position": (0.5, 0.5),
            "refinement_extent": (-1.5, 1.5, -1.5, 1.5),
        },
    )

    data, extent = _valid_data_and_extent(block)

    assert data.shape == (6, 6)
    assert data.tolist() == [
        [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
        [19.0, 20.0, 21.0, 22.0, 23.0, 24.0],
        [28.0, 29.0, 30.0, 31.0, 32.0, 33.0],
        [37.0, 38.0, 39.0, 40.0, 41.0, 42.0],
        [46.0, 47.0, 48.0, 49.0, 50.0, 51.0],
        [55.0, 56.0, 57.0, 58.0, 59.0, 60.0],
    ]
    assert extent == (-1.5, 1.5, -1.5, 1.5)


def test_valid_data_and_extent_respects_vertex_centered_position() -> None:
    block = GridBlock(
        data=np.arange(25, dtype=float).reshape(5, 5),
        axes=("z", "x"),
        origin=(-2.0, -2.0),
        spacing=(1.0, 1.0),
        metadata={
            "grid_position": (0.0, 0.0),
            "refinement_extent": (-1.0, 2.0, -1.0, 2.0),
        },
    )

    data, extent = _valid_data_and_extent(block)

    assert data.shape == (4, 4)
    assert data.tolist() == [
        [6.0, 7.0, 8.0, 9.0],
        [11.0, 12.0, 13.0, 14.0],
        [16.0, 17.0, 18.0, 19.0],
        [21.0, 22.0, 23.0, 24.0],
    ]
    assert extent == (-1.0, 2.0, -1.0, 2.0)
