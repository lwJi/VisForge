from __future__ import annotations

from pathlib import Path

import pytest

from visforge.data.model import PlaneSpec
from visforge.cli import _build_parser, _slice_options
from visforge.plotting.base import PlotLabels


def test_plot_slice_mesh_defaults() -> None:
    parser = _build_parser()
    args = parser.parse_args(
        [
            "plot-slice",
            "dataset",
            "--field",
            "gfc",
            "--output",
            "gfc.png",
        ]
    )
    options = _slice_options(args)
    assert options["mesh_linewidth"] == 0.15
    assert options["mesh_alpha"] == 0.75
    assert options["component"] is None
    assert options["cmap"] == "viridis"
    assert options["scale"] == "linear"
    assert options["vmin"] is None
    assert options["vmax"] is None
    assert args.xlim is None
    assert args.ylim is None


def test_plot_slice_component_option() -> None:
    parser = _build_parser()
    args = parser.parse_args(
        [
            "plot-slice",
            "dataset",
            "--field",
            "state",
            "--component",
            "testsubcyclingmc2_rho",
            "--output",
            "rho.png",
        ]
    )
    options = _slice_options(args)
    assert options["field"] == "state"
    assert options["component"] == "testsubcyclingmc2_rho"


def test_plot_slice_axis_range_options() -> None:
    parser = _build_parser()
    args = parser.parse_args(
        [
            "plot-slice",
            "dataset",
            "--field",
            "gfc",
            "--xlim",
            "-2",
            "2",
            "--ylim",
            "-1.5",
            "1.5",
            "--output",
            "gfc.png",
        ]
    )
    assert args.xlim == [-2.0, 2.0]
    assert args.ylim == [-1.5, 1.5]


def test_plot_slice_color_range_options() -> None:
    parser = _build_parser()
    args = parser.parse_args(
        [
            "plot-slice",
            "dataset",
            "--field",
            "state",
            "--component",
            "testsubcyclingmc2_rho",
            "--vmin",
            "0",
            "--vmax",
            "1.5",
            "--output",
            "rho.png",
        ]
    )
    options = _slice_options(args)
    assert options["vmin"] == 0.0
    assert options["vmax"] == 1.5


def test_plot_slice_scale_option() -> None:
    parser = _build_parser()
    args = parser.parse_args(
        [
            "plot-slice",
            "dataset",
            "--field",
            "state",
            "--scale",
            "log",
            "--output",
            "rho.png",
        ]
    )
    options = _slice_options(args)
    assert options["scale"] == "log"


def test_plot_slice_colormap_option() -> None:
    parser = _build_parser()
    args = parser.parse_args(
        [
            "plot-slice",
            "dataset",
            "--field",
            "state",
            "--cmap",
            "plasma",
            "--output",
            "rho.png",
        ]
    )
    options = _slice_options(args)
    assert options["cmap"] == "plasma"


def test_plot_line_label_options() -> None:
    parser = _build_parser()
    args = parser.parse_args(
        [
            "plot-line",
            "dataset",
            "--field",
            "gfc",
            "--axis",
            "x",
            "--title",
            "Line title",
            "--xlabel",
            "Radius",
            "--ylabel",
            "Density",
            "--legend",
            "gfc",
            "--output",
            "gfc.png",
        ]
    )

    assert args.title == "Line title"
    assert args.xlabel == "Radius"
    assert args.ylabel == "Density"
    assert args.legend_label == "gfc"


def test_plot_slice_label_options() -> None:
    parser = _build_parser()
    args = parser.parse_args(
        [
            "plot-slice",
            "dataset",
            "--field",
            "gfc",
            "--title",
            "Slice title",
            "--xlabel",
            "Radius",
            "--ylabel",
            "Height",
            "--colorbar",
            "Density",
            "--output",
            "gfc.png",
        ]
    )
    options = _slice_options(args)

    assert options["labels"] == PlotLabels(
        title="Slice title",
        xlabel="Radius",
        ylabel="Height",
        colorbar="Density",
    )


def test_plot_slice_rejects_unknown_colormap() -> None:
    parser = _build_parser()
    args = parser.parse_args(
        [
            "plot-slice",
            "dataset",
            "--field",
            "state",
            "--cmap",
            "not_a_cmap",
            "--output",
            "rho.png",
        ]
    )
    with pytest.raises(ValueError, match="Unknown colormap 'not_a_cmap'"):
        _slice_options(args)


def test_plot_slice_config_file_supplies_defaults(tmp_path) -> None:
    config = tmp_path / "plot.yaml"
    config.write_text(
        """
dataset: /data/run
plot:
  field: gfc
  component: gfc
  iteration: 0
  plane: xz
  backend: openpmd
  cmap: cividis
  vmin: -1
  vmax: 1
  output: gfc.png
mesh:
  show: true
  color: cyan
  linewidth: 0.2
  alpha: 0.4
view:
  xlim: [-4, 4]
  ylim: [-3, 3]
labels:
  title: Density slice
  xlabel: Radius
  ylabel: Height
  colorbar: Density
""",
        encoding="utf-8",
    )
    parser = _build_parser()
    args = parser.parse_args(["plot-slice", "--config", str(config)])
    options = _slice_options(args)
    assert options["path"] == Path("/data/run")
    assert options["field"] == "gfc"
    assert options["component"] == "gfc"
    assert options["iteration"] == 0
    assert options["plane"] == "xz"
    assert options["backend"] == "openpmd"
    assert options["cmap"] == "cividis"
    assert options["show_mesh"] is True
    assert options["mesh_color"] == "cyan"
    assert options["mesh_linewidth"] == 0.2
    assert options["mesh_alpha"] == 0.4
    assert options["vmin"] == -1.0
    assert options["vmax"] == 1.0
    assert options["xlim"] == (-4.0, 4.0)
    assert options["ylim"] == (-3.0, 3.0)
    assert options["labels"] == PlotLabels(
        title="Density slice",
        xlabel="Radius",
        ylabel="Height",
        colorbar="Density",
    )


def test_plot_slice_config_file_supplies_scale(tmp_path) -> None:
    config = tmp_path / "plot.yaml"
    config.write_text(
        """
dataset: /data/run
plot:
  field: rho
  scale: log
  output: rho.png
""",
        encoding="utf-8",
    )
    parser = _build_parser()
    args = parser.parse_args(["plot-slice", "--config", str(config)])
    options = _slice_options(args)
    assert options["scale"] == "log"


def test_plot_slice_cli_overrides_config(tmp_path) -> None:
    config = tmp_path / "plot.yaml"
    config.write_text(
        """
dataset: /data/run
plot:
  field: gfc
  output: gfc.png
view:
  xlim: [-4, 4]
labels:
  title: Config title
  xlabel: Config x
""",
        encoding="utf-8",
    )
    parser = _build_parser()
    args = parser.parse_args(
        [
            "plot-slice",
            "--config",
            str(config),
            "--field",
            "gfv",
            "--title",
            "CLI title",
            "--xlim",
            "-1",
            "1",
        ]
    )
    options = _slice_options(args)
    assert options["field"] == "gfv"
    assert options["xlim"] == (-1.0, 1.0)
    assert options["labels"] == PlotLabels(title="CLI title", xlabel="Config x")


def test_plot_slice_config_file_supplies_sample_plane(tmp_path) -> None:
    config = tmp_path / "plot.yaml"
    config.write_text(
        """
dataset: /data/run
plot:
  field: gfc
  output: gfc.png
sample_plane:
  origin: [0, 0, 0]
  normal: [1, 1, 0]
  up: [0, 0, 1]
  size: [8, 6]
  resolution: [512, 384]
  interpolation: nearest
""",
        encoding="utf-8",
    )
    parser = _build_parser()
    args = parser.parse_args(["plot-slice", "--config", str(config)])
    options = _slice_options(args)
    assert options["sample_plane"] == PlaneSpec(
        origin=(0.0, 0.0, 0.0),
        normal=(1.0, 1.0, 0.0),
        up=(0.0, 0.0, 1.0),
        size=(8.0, 6.0),
        resolution=(512, 384),
        interpolation="nearest",
    )


def test_plot_slice_cli_supplies_sample_plane() -> None:
    parser = _build_parser()
    args = parser.parse_args(
        [
            "plot-slice",
            "dataset",
            "--field",
            "gfc",
            "--sample-plane-origin",
            "0",
            "0",
            "0",
            "--sample-plane-normal",
            "1",
            "0",
            "0",
            "--sample-plane-up",
            "0",
            "0",
            "1",
            "--sample-plane-size",
            "4",
            "4",
            "--sample-plane-resolution",
            "128",
            "128",
            "--interpolation",
            "linear",
            "--output",
            "gfc.png",
        ]
    )
    options = _slice_options(args)
    assert options["sample_plane"] == PlaneSpec(
        origin=(0.0, 0.0, 0.0),
        normal=(1.0, 0.0, 0.0),
        up=(0.0, 0.0, 1.0),
        size=(4.0, 4.0),
        resolution=(128, 128),
        interpolation="linear",
    )


def test_plot_slice_rejects_plane_and_sample_plane(tmp_path) -> None:
    config = tmp_path / "plot.yaml"
    config.write_text(
        """
dataset: /data/run
plot:
  field: gfc
  plane: xy
  output: gfc.png
sample_plane:
  origin: [0, 0, 0]
  normal: [0, 0, 1]
  up: [0, 1, 0]
  size: [1, 1]
  resolution: [32, 32]
""",
        encoding="utf-8",
    )
    parser = _build_parser()
    args = parser.parse_args(["plot-slice", "--config", str(config)])
    with pytest.raises(SystemExit, match="either --plane or sample_plane"):
        _slice_options(args)
