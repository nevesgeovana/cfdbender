"""Record the strip integration of another library on seeded inputs.

Run once with ``PYTHONPATH=src`` and the source folder of the library that defines the
integrated sectional-load columns on the path as well. The outputs go to
``tests/data/section_strips_reference.json``, which the test suite compares with
:func:`cfdbender.post.sections.strip_lengths` exactly.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from pyflightstream.post._tables import _integrated_strips  # type: ignore[import-not-found]


def main() -> None:
    """Write the reference file."""
    rng = random.Random(20260518)
    cases = []
    for index in range(12):
        n = rng.randint(2, 40)
        rising = index % 2 == 0
        offsets = [0.0]
        for _ in range(n - 1):
            offsets.append(offsets[-1] + rng.uniform(0.01, 0.5))
        if not rising:
            offsets = offsets[::-1]
        rows = [[offsets[i], *(rng.uniform(-300.0, 300.0) for _ in range(6))] for i in range(n)]
        out = _integrated_strips([list(r) for r in rows])
        cases.append({"rows": rows, "strips": [list(o) for o in out]})
    target = (
        Path(__file__).resolve().parents[1] / "tests" / "data" / "section_strips_reference.json"
    )
    target.write_text(json.dumps({"cases": cases}, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {target} with {len(cases)} cases")


if __name__ == "__main__":
    main()
