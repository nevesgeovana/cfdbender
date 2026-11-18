"""Review regressions for the bounded SU2 marker scan."""

import pytest

from cfdbender.solvers._host import _groups


@pytest.mark.parametrize("length", [19, 20, 21])
def test_complete_relevant_line_obeys_long_line_limit(monkeypatch, length):
    monkeypatch.setattr(_groups, "_LONG_LINE", 20)
    data = b"MARKER_TAG=" + b"x" * (length - len(b"MARKER_TAG=")) + b"\n"
    scanner = _groups._Su2Markers()
    scanner.feed(data)
    result = scanner.result()
    if length > 20:
        assert result is None
    else:
        assert result == _groups.marker_names(data, "m.su2")


def test_reduced_limit_and_exact_fallback(tmp_path, monkeypatch):
    from cfdbender.core import plan as plan_module
    from cfdbender.core.plan import InputFiles

    monkeypatch.setattr(plan_module, "LARGE_INPUT_BYTES", 1)
    monkeypatch.setattr(_groups, "_REDUCED_LIMIT", 20)
    exact = b"MARKER_TAG=" + b"x" * 8
    scanner = _groups._Su2Markers()
    scanner.feed(exact)
    assert scanner.result() == _groups.marker_names(exact, "m.su2")
    data = exact + b"xx"
    scanner = _groups._Su2Markers()
    scanner.feed(data)
    assert scanner.result() is None
    (tmp_path / "m.su2").write_bytes(data)
    calls = []
    original = _groups.marker_names

    def observed(content, name):
        calls.append(content)
        return original(content, name)

    monkeypatch.setattr(_groups, "marker_names", observed)
    assert _groups.tracked_markers(InputFiles(tmp_path.resolve()), "m.su2") == original(
        data, "m.su2"
    )
    assert calls == [data]


def test_tag_after_long_whitespace_prefix_falls_back(tmp_path, monkeypatch):
    from cfdbender.core import plan as plan_module
    from cfdbender.core.plan import InputFiles

    monkeypatch.setattr(plan_module, "LARGE_INPUT_BYTES", 1)
    monkeypatch.setattr(plan_module, "_BLOCK", 8)
    monkeypatch.setattr(_groups, "_LONG_LINE", 20)
    data = b" " * 21 + b"MARKER_TAG= wall\n"
    (tmp_path / "m.su2").write_bytes(data)
    assert _groups.tracked_markers(InputFiles(tmp_path.resolve()), "m.su2") == ("wall",)
    assert _groups.marker_names(data, "m.su2") == ("wall",)


def test_mutated_byte_streams_match_whole_file():
    import random

    rng = random.Random(981736)
    seeds = [
        b"MARKER_TAG\n=\nwall\n",
        b"MARKER_TAG=\n \n%\nfoo\nMARKER_TAG=two\n",
        b"  MARKER_TAG=\xff\x00\r\n",
        b"MARKER_TAG=\n\x0c\nend\n",
    ]
    atoms = [
        b"\n",
        b"\r",
        b"\x0b",
        b"\x0c",
        b"\x1c",
        b"\x00",
        b"\xff",
        b" ",
        b"%",
        b"=",
        b"MARKER_TAG",
        b"marker_tag",
        b"x",
        b"\t",
    ]
    count = 0
    for _ in range(8000):
        data = rng.choice(seeds)
        for _ in range(rng.randrange(1, 9)):
            start = rng.randrange(len(data) + 1)
            end = min(len(data), start + rng.randrange(0, 5))
            data = data[:start] + rng.choice(atoms) + data[end:]
        expected = _groups.marker_names(data, "m.su2")
        for block in (1, 2, 3, 5, 7, 64, 1 << 20):
            scanner = _groups._Su2Markers()
            for start in range(0, len(data), block):
                scanner.feed(data[start : start + block])
            assert scanner.result() == expected, (data, block)
            count += 1
    assert count == 56000


def test_reduced_text_limit_counts_actual_joined_bytes(monkeypatch):
    monkeypatch.setattr(_groups, "_REDUCED_LIMIT", 20)
    for size in (19, 20, 21):
        data = b"MARKER_TAG=" + b"x" * (size - len(b"MARKER_TAG="))
        scanner = _groups._Su2Markers()
        scanner.feed(data)
        if size <= 20:
            assert scanner.result() == _groups.marker_names(data, "m.su2")
        else:
            assert scanner.result() is None
