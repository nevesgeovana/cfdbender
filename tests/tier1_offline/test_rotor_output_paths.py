"""Rotor table names cannot turn a reference-table alias into a path."""

from __future__ import annotations

import pytest

from cfdbender.matrix._rotor_table import rotor_table


@pytest.mark.parametrize("alias", ["x/../../../../../escape", "x\\..\\escape", "x\nname"])
def test_rotor_alias_cannot_escape_product_directory(tmp_path, alias):
    writes = []
    rows = [{"pol": "1", "result": {"rotor": {"rotor": alias, "row": {"CT": 1.0}}}}]

    def write(path, text):
        writes.append(path)
        return path

    with pytest.raises(ValueError):
        rotor_table(tmp_path, rows, write=write, cell=str)
    assert writes == []


@pytest.mark.parametrize("product", ["probe", "volume", "surface", "distribution"])
def test_post_product_labels_are_not_paths(tmp_path, product):
    import numpy as np
    from post_sections_support import plate
    from post_volume_support import box_grid

    from cfdbender.post.sections import (
        Distribution,
        Reference,
        compute_sections,
        write_section_files,
    )
    from cfdbender.post.volume_files import (
        ProbeEntry,
        VolumeStep,
        write_probe_table,
        write_volume_section,
    )
    from cfdbender.post.volume_frames import Frame

    # All attempted escapes remain inside pytest's disposable directory.
    directory = tmp_path / "products"
    point = "../../escaped"
    points = np.array([[0.5, 0.5, 0.5]])
    if product == "distribution":
        for prefix in ("point_cp_x", "point_sloads_x"):
            (directory / "sections" / prefix).mkdir(parents=True)
    with pytest.raises(ValueError):
        if product == "probe":
            write_probe_table(directory, point, [ProbeEntry(points, Frame())], [], {})
        elif product == "volume":
            write_volume_section(
                directory,
                point,
                points,
                Frame(),
                [VolumeStep(None, None, box_grid("hex", (1, 1, 1)))],
                {},
            )
        else:
            result = compute_sections(plate(2, 3, 4, 6), Reference(0, 1), Distribution(count=1))
            write_section_files(
                result,
                directory,
                "point" if product == "distribution" else point,
                "x/../../../escaped" if product == "distribution" else "cut",
            )
    assert not list(tmp_path.glob("escaped*"))
