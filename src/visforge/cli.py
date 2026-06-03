"""Command-line interface for VisForge."""

from __future__ import annotations

import argparse
from pathlib import Path

from visforge.workflows.inspect import format_summary, inspect_dataset
from visforge.workflows.line_plot import make_line_plot
from visforge.workflows.slice_plot import make_slice_plot


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="visforge")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect", help="Summarize a CarpetX dataset")
    inspect_parser.add_argument("path", type=Path)
    inspect_parser.add_argument("--backend", default="auto", choices=("auto", "openpmd", "silo", "tsv"))
    inspect_parser.set_defaults(func=_inspect)

    line_parser = subparsers.add_parser("plot-line", help="Plot a CarpetX TSV line output")
    line_parser.add_argument("path", type=Path)
    line_parser.add_argument("--field", required=True)
    line_parser.add_argument("--axis", required=True, choices=("x", "y", "z"))
    line_parser.add_argument("--iteration", type=int)
    line_parser.add_argument("--backend", default="tsv", choices=("auto", "tsv"))
    line_parser.add_argument("--output", required=True, type=Path)
    line_parser.set_defaults(func=_plot_line)

    slice_parser = subparsers.add_parser("plot-slice", help="Plot an openPMD scalar slice")
    slice_parser.add_argument("path", type=Path)
    slice_parser.add_argument("--field", required=True)
    slice_parser.add_argument("--iteration", type=int)
    slice_parser.add_argument("--plane", choices=("xy", "xz", "yz"))
    slice_parser.add_argument("--backend", default="auto", choices=("auto", "openpmd", "silo"))
    slice_parser.add_argument(
        "--show-mesh",
        action="store_true",
        help="Overlay cell-boundary mesh lines and block outlines on the scalar slice plot",
    )
    slice_parser.add_argument("--mesh-color", default="white", help="Mesh overlay color")
    slice_parser.add_argument(
        "--mesh-linewidth",
        default=0.8,
        type=float,
        help="Mesh overlay line width in points",
    )
    slice_parser.add_argument(
        "--mesh-max-lines",
        type=int,
        help="Decimate mesh overlay to at most this many lines per axis and block",
    )
    slice_parser.add_argument("--output", required=True, type=Path)
    slice_parser.set_defaults(func=_plot_slice)
    return parser


def _inspect(args: argparse.Namespace) -> int:
    print(format_summary(inspect_dataset(args.path, backend=args.backend)))
    return 0


def _plot_line(args: argparse.Namespace) -> int:
    result = make_line_plot(
        args.path,
        field=args.field,
        axis=args.axis,
        iteration=args.iteration,
        backend=args.backend,
        output=args.output,
    )
    print(f"Wrote {result.output}")
    return 0


def _plot_slice(args: argparse.Namespace) -> int:
    result = make_slice_plot(
        args.path,
        field=args.field,
        iteration=args.iteration,
        plane=args.plane,
        backend=args.backend,
        output=args.output,
        show_mesh=args.show_mesh,
        mesh_color=args.mesh_color,
        mesh_linewidth=args.mesh_linewidth,
        mesh_max_lines=args.mesh_max_lines,
    )
    print(f"Wrote {result.output}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
