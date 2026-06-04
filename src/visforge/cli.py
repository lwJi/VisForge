"""Command-line interface for VisForge."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from visforge.config import bool_value, load_config, range_value, section
from visforge.data.model import PlaneSpec
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
    slice_parser.add_argument("path", nargs="?", type=Path)
    slice_parser.add_argument("--config", type=Path, help="YAML config file with plot-slice defaults")
    slice_parser.add_argument("--field")
    slice_parser.add_argument("--component", help="openPMD record component to plot when a field has multiple components")
    slice_parser.add_argument("--iteration", type=int)
    slice_parser.add_argument("--plane", choices=("xy", "xz", "yz"))
    slice_parser.add_argument(
        "--sample-plane-origin",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        help="Center of a user-defined 2D plane sampled from a 3D field",
    )
    slice_parser.add_argument(
        "--sample-plane-normal",
        nargs=3,
        type=float,
        metavar=("NX", "NY", "NZ"),
        help="Normal vector of a user-defined 2D plane",
    )
    slice_parser.add_argument(
        "--sample-plane-up",
        nargs=3,
        type=float,
        metavar=("UX", "UY", "UZ"),
        help="Preferred vertical direction for a user-defined 2D plane",
    )
    slice_parser.add_argument(
        "--sample-plane-size",
        nargs=2,
        type=float,
        metavar=("WIDTH", "HEIGHT"),
        help="Physical width and height of a user-defined 2D plane",
    )
    slice_parser.add_argument(
        "--sample-plane-resolution",
        nargs=2,
        type=int,
        metavar=("NX", "NY"),
        help="Output sample resolution for a user-defined 2D plane",
    )
    slice_parser.add_argument(
        "--interpolation",
        choices=("linear", "nearest"),
        help="Interpolation method for a user-defined 2D plane",
    )
    slice_parser.add_argument("--backend", default="auto", choices=("auto", "openpmd", "silo"))
    slice_parser.add_argument(
        "--show-mesh",
        action="store_true",
        help="Overlay cell-boundary mesh lines and block outlines on the scalar slice plot",
    )
    slice_parser.add_argument("--mesh-color", default="white", help="Mesh overlay color")
    slice_parser.add_argument(
        "--mesh-linewidth",
        default=None,
        type=float,
        help="Mesh overlay line width in points",
    )
    slice_parser.add_argument(
        "--mesh-alpha",
        default=None,
        type=float,
        help="Mesh overlay opacity from 0.0 to 1.0",
    )
    slice_parser.add_argument(
        "--mesh-max-lines",
        type=int,
        help="Decimate mesh overlay to at most this many lines per axis and block",
    )
    slice_parser.add_argument(
        "--xlim",
        nargs=2,
        type=float,
        metavar=("XMIN", "XMAX"),
        help="Visible x-axis range",
    )
    slice_parser.add_argument(
        "--ylim",
        nargs=2,
        type=float,
        metavar=("YMIN", "YMAX"),
        help="Visible y-axis range",
    )
    slice_parser.add_argument("--vmin", type=float, help="Minimum scalar value for the color range")
    slice_parser.add_argument("--vmax", type=float, help="Maximum scalar value for the color range")
    slice_parser.add_argument("--output", type=Path)
    slice_parser.set_defaults(func=_plot_slice)
    return parser


def _inspect(args: argparse.Namespace) -> int:
    print(format_summary(inspect_dataset(args.path, backend=args.backend)))
    return 0


def _plot_line(args: argparse.Namespace) -> int:
    try:
        result = make_line_plot(
            args.path,
            field=args.field,
            axis=args.axis,
            iteration=args.iteration,
            backend=args.backend,
            output=args.output,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from None
    print(f"Wrote {result.output}")
    return 0


def _plot_slice(args: argparse.Namespace) -> int:
    options = _slice_options(args)
    try:
        result = make_slice_plot(
            options["path"],
            field=options["field"],
            component=options["component"],
            iteration=options["iteration"],
            plane=options["plane"],
            sample_plane=options["sample_plane"],
            backend=options["backend"],
            output=options["output"],
            show_mesh=options["show_mesh"],
            mesh_color=options["mesh_color"],
            mesh_linewidth=options["mesh_linewidth"],
            mesh_alpha=options["mesh_alpha"],
            mesh_max_lines=options["mesh_max_lines"],
            xlim=options["xlim"],
            ylim=options["ylim"],
            vmin=options["vmin"],
            vmax=options["vmax"],
        )
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from None
    print(f"Wrote {result.output}")
    return 0


def _slice_options(args: argparse.Namespace) -> dict[str, object]:
    config = load_config(args.config)
    plot = section(config, "plot")
    mesh = section(config, "mesh")
    view = section(config, "view")
    sample = section(config, "sample_plane")

    options = {
        "path": args.path or config.get("dataset") or plot.get("dataset"),
        "field": _choose(args.field, plot.get("field")),
        "component": _choose(args.component, plot.get("component")),
        "iteration": _choose(args.iteration, plot.get("iteration")),
        "plane": _choose(args.plane, plot.get("plane")),
        "sample_plane": _sample_plane(args, sample),
        "backend": _choose(args.backend if args.backend != "auto" else None, plot.get("backend"), "auto"),
        "output": _choose(args.output, plot.get("output")),
        "show_mesh": bool_value(_choose(args.show_mesh if args.show_mesh else None, mesh.get("show"), False)),
        "mesh_color": _choose(args.mesh_color if args.mesh_color != "white" else None, mesh.get("color"), "white"),
        "mesh_linewidth": float(_choose(args.mesh_linewidth, mesh.get("linewidth"), 0.15)),
        "mesh_alpha": float(_choose(args.mesh_alpha, mesh.get("alpha"), 0.75)),
        "mesh_max_lines": _choose(args.mesh_max_lines, mesh.get("max_lines")),
        "xlim": _choose(_range_tuple(args.xlim), range_value(view.get("xlim"))),
        "ylim": _choose(_range_tuple(args.ylim), range_value(view.get("ylim"))),
        "vmin": _choose(args.vmin, plot.get("vmin")),
        "vmax": _choose(args.vmax, plot.get("vmax")),
    }
    missing = [name for name in ("path", "field", "output") if options[name] is None]
    if missing:
        raise SystemExit(f"plot-slice requires {', '.join(missing)} from CLI or config.")
    if options["iteration"] is not None:
        options["iteration"] = int(options["iteration"])
    if options["mesh_max_lines"] is not None:
        options["mesh_max_lines"] = int(options["mesh_max_lines"])
    if options["vmin"] is not None:
        options["vmin"] = float(options["vmin"])
    if options["vmax"] is not None:
        options["vmax"] = float(options["vmax"])
    if options["plane"] is not None and options["sample_plane"] is not None:
        raise SystemExit("plot-slice accepts either --plane or sample_plane config, not both.")
    if options["path"] is not None:
        options["path"] = Path(options["path"])
    if options["output"] is not None:
        options["output"] = Path(options["output"])
    return options


def _choose(*values):
    for value in values:
        if value is not None:
            return value
    return None


def _range_tuple(values: list[float] | None) -> tuple[float, float] | None:
    if values is None:
        return None
    start, stop = values
    return start, stop


def _sample_plane(args: argparse.Namespace, config: dict[str, Any]) -> PlaneSpec | None:
    cli_values = (
        args.sample_plane_origin,
        args.sample_plane_normal,
        args.sample_plane_up,
        args.sample_plane_size,
        args.sample_plane_resolution,
        args.interpolation,
    )
    if not config and all(value is None for value in cli_values):
        return None

    origin = _choose(_vector_tuple(args.sample_plane_origin, 3), _vector_tuple(config.get("origin"), 3))
    normal = _choose(_vector_tuple(args.sample_plane_normal, 3), _vector_tuple(config.get("normal"), 3))
    up = _choose(_vector_tuple(args.sample_plane_up, 3), _vector_tuple(config.get("up"), 3))
    size = _choose(_vector_tuple(args.sample_plane_size, 2), _vector_tuple(config.get("size"), 2))
    resolution = _choose(
        _int_tuple(args.sample_plane_resolution, 2),
        _int_tuple(config.get("resolution"), 2),
    )
    interpolation = _choose(args.interpolation, config.get("interpolation"), "linear")
    missing = [
        name
        for name, value in (
            ("origin", origin),
            ("normal", normal),
            ("up", up),
            ("size", size),
            ("resolution", resolution),
        )
        if value is None
    ]
    if missing:
        raise SystemExit(f"sample_plane requires {', '.join(missing)}.")
    if interpolation not in {"linear", "nearest"}:
        raise SystemExit("sample_plane.interpolation must be 'linear' or 'nearest'.")
    return PlaneSpec(
        origin=origin,
        normal=normal,
        up=up,
        size=size,
        resolution=resolution,
        interpolation=interpolation,
    )


def _vector_tuple(values: Any, length: int) -> tuple[float, ...] | None:
    if values is None:
        return None
    if not isinstance(values, (list, tuple)) or len(values) != length:
        raise ValueError(f"Expected {length} numeric values.")
    return tuple(float(value) for value in values)


def _int_tuple(values: Any, length: int) -> tuple[int, ...] | None:
    if values is None:
        return None
    if not isinstance(values, (list, tuple)) or len(values) != length:
        raise ValueError(f"Expected {length} integer values.")
    return tuple(int(value) for value in values)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
