"""Polar product entries retain the identity needed by schema-version-two readers."""

from cfdbender.matrix.products import _entry


def test_text_polar_product_names_its_polar(tmp_path):
    path = tmp_path / "post/study/polars/P1001.txt"
    path.parent.mkdir(parents=True)
    path.write_text("polar\n", encoding="utf-8")
    entry = _entry(tmp_path, path)
    assert entry["kind"] == "polar"
    assert entry.get("polar") == "1001"
