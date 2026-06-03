"""Plot a CarpetX TSV line output."""

from __future__ import annotations

import argparse

from visforge.workflows.line_plot import make_line_plot


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("--field", required=True)
    parser.add_argument("--axis", required=True, choices=("x", "y", "z"))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    make_line_plot(args.path, field=args.field, axis=args.axis, output=args.output)


if __name__ == "__main__":
    main()
