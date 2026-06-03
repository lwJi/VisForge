"""Inspect a CarpetX output tree."""

from __future__ import annotations

import argparse

from visforge.workflows.inspect import format_summary, inspect_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    args = parser.parse_args()
    print(format_summary(inspect_dataset(args.path)))


if __name__ == "__main__":
    main()
