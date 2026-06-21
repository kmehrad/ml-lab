from pathlib import Path

import pytest

from src.download_arxiv_papers import arxiv_id_from_url, download_pdf


class FakeResponse:
    def __init__(self, content: bytes) -> None:
        self.content = content
        self.position = 0

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self, size: int) -> bytes:
        chunk = self.content[self.position : self.position + size]
        self.position += len(chunk)
        return chunk


def test_arxiv_id_from_url_handles_pdf_suffix() -> None:
    assert (
        arxiv_id_from_url("https://arxiv.org/pdf/2606.20497.pdf")
        == "2606.20497"
    )


def test_download_pdf_writes_valid_pdf(tmp_path: Path) -> None:
    content = b"%PDF-1.7\nexample"

    def fake_opener(*args: object, **kwargs: object) -> FakeResponse:
        return FakeResponse(content)

    path = download_pdf(
        "https://arxiv.org/pdf/2606.20497",
        tmp_path,
        opener=fake_opener,
    )

    assert path == tmp_path / "2606.20497.pdf"
    assert path.read_bytes() == content


def test_download_pdf_rejects_non_pdf_response(tmp_path: Path) -> None:
    def fake_opener(*args: object, **kwargs: object) -> FakeResponse:
        return FakeResponse(b"<html>Not a PDF</html>")

    with pytest.raises(ValueError, match="not a PDF"):
        download_pdf(
            "https://arxiv.org/pdf/2606.20497",
            tmp_path,
            opener=fake_opener,
        )

    assert not (tmp_path / "2606.20497.pdf").exists()
    assert not (tmp_path / "2606.20497.pdf.part").exists()
