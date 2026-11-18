"""The strip integration equals the recorded outputs of the library whose columns it follows.

The reference file is written once by ``tools/record_pyfs_strips.py``; the comparison is exact.
"""

from __future__ import annotations

import json
from pathlib import Path

from cfdbender.post.sections import strip_lengths

DATA = Path(__file__).resolve().parents[1] / "data" / "section_strips_reference.json"


def test_strip_lengths_and_integrated_loads_equal_the_recorded_reference_exactly():
    cases = json.loads(DATA.read_text(encoding="utf-8"))["cases"]
    assert len(cases) == 12
    for case in cases:
        rows = case["rows"]
        lengths = strip_lengths([r[0] for r in rows])
        mine = [
            [length, *(r[i] * length for i in (4, 5, 6))]
            for r, length in zip(rows, lengths, strict=True)
        ]
        assert mine == case["strips"]
