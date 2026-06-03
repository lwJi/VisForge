from __future__ import annotations

from pathlib import Path

from visforge.cli import _build_parser, _slice_options


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
    assert args.xlim is None
    assert args.ylim is None


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


def test_plot_slice_config_file_supplies_defaults(tmp_path) -> None:
    config = tmp_path / "plot.yaml"
    config.write_text(
        """
dataset: /data/run
plot:
  field: gfc
  iteration: 0
  plane: xz
  backend: openpmd
  output: gfc.png
mesh:
  show: true
  color: cyan
  linewidth: 0.2
  alpha: 0.4
view:
  xlim: [-4, 4]
  ylim: [-3, 3]
""",
        encoding="utf-8",
    )
    parser = _build_parser()
    args = parser.parse_args(["plot-slice", "--config", str(config)])
    options = _slice_options(args)
    assert options["path"] == Path("/data/run")
    assert options["field"] == "gfc"
    assert options["iteration"] == 0
    assert options["plane"] == "xz"
    assert options["backend"] == "openpmd"
    assert options["show_mesh"] is True
    assert options["mesh_color"] == "cyan"
    assert options["mesh_linewidth"] == 0.2
    assert options["mesh_alpha"] == 0.4
    assert options["xlim"] == (-4.0, 4.0)
    assert options["ylim"] == (-3.0, 3.0)


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
            "--xlim",
            "-1",
            "1",
        ]
    )
    options = _slice_options(args)
    assert options["field"] == "gfv"
    assert options["xlim"] == (-1.0, 1.0)
