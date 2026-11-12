"""Run the measured SU2 8.6 serial lifecycle in an expendable interpreter.

SU2 can terminate the process on an input error. Only the worker imports the
binding; the parent captures its channels and classifies saved numerical data.
"""

from __future__ import annotations

import argparse
import importlib


def main() -> None:
    """Parse the cfg path and run the reference launcher's serial lifecycle."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    args = parser.parse_args()
    pysu2 = importlib.import_module("pysu2")
    driver = pysu2.CSinglezoneDriver(args.config, 1, 0)
    driver.StartSolver()
    driver.Finalize()


if __name__ == "__main__":
    main()
