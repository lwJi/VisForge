from __future__ import annotations

from visforge.cli import _build_parser


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
    assert args.mesh_linewidth == 0.15
    assert args.mesh_alpha == 0.75
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
