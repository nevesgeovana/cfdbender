"""R2-02: restore must validate the same tree that it copies."""

import pytest
from test_p0000_item3c_adapter import workspace
from test_review_paths import link_folder

from cfdbender.matrix.cli import main


@pytest.mark.parametrize("apply", [False, True])
def test_restore_refuses_a_linked_descendant_before_replacing_products(tmp_path, apply):
    root = tmp_path / "workspace"
    workspace(root)
    source = root / "archive/old/post/study"
    source.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "private.txt").write_text("must not be copied")
    link_folder(source / "nested", outside)
    live = root / "post/study/keep.txt"
    live.parent.mkdir(parents=True)
    live.write_text("current products")
    args = ["restore", "products", "--matrix", "study", "--workspace", str(root)]
    assert main([*args, *(["--apply"] if apply else [])]) == 2
    assert live.read_text() == "current products"
    assert not (live.parent / "nested/private.txt").exists()
