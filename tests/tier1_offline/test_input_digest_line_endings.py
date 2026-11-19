from pathlib import Path

from cfdbender.core.plan import InputFiles


def test_input_digest_documents_line_ending_sensitivity(tmp_path):
    source = tmp_path / "input.txt"
    source.write_bytes(b"a\nb\n")
    lf = InputFiles(tmp_path).digest("input.txt")
    source.write_bytes(b"a\r\nb\r\n")
    crlf = InputFiles(tmp_path).digest("input.txt")
    assert lf != crlf
    docs = (Path(__file__).resolve().parents[2] / "docs/workspace-layout.md").read_text(
        encoding="utf-8"
    )
    assert "LF and CRLF" in docs and "raw bytes" in docs
