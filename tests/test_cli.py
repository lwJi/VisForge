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
