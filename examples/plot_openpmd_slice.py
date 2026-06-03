"""Plot a scalar slice from CarpetX openPMD output."""

from __future__ import annotations

import argparse

from visforge.workflows.slice_plot import make_slice_plot


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("--field", required=True)
    parser.add_argument("--iteration", type=int)
    parser.add_argument("--plane", choices=("xy", "xz", "yz"))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    make_slice_plot(
        args.path,
        field=args.field,
        iteration=args.iteration,
        plane=args.plane,
        backend="openpmd",
        output=args.output,
    )


if __name__ == "__main__":
    main()
