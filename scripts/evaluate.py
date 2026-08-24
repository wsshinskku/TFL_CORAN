"""Print a run's generated evaluation summary without hard-coded values."""

from __future__ import annotations

import argparse

from tfl_coran.cli import command_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    command_report(args)


if __name__ == "__main__":
    main()
