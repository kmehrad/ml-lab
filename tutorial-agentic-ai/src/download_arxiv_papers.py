"""Download the arXiv papers used by the RAG tutorials."""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

ARXIV_PDF_URLS = (
    "https://arxiv.org/pdf/2606.20497",
    "https://arxiv.org/pdf/2606.20563",
    "https://arxiv.org/pdf/2606.20539",
    "https://arxiv.org/pdf/2606.20331",
    "https://arxiv.org/pdf/2606.20374",
)

DEFAULT_OUTPUT_DIR = Path("data/raw/arxiv")
CHUNK_SIZE = 1024 * 1024
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
USER_AGENT = "tutorial-agentic-ai/0.1 (educational RAG example)"


def arxiv_id_from_url(url: str) -> str:
    """Extract an arXiv identifier from a PDF URL."""
    arxiv_id = Path(urlparse(url).path).name.removesuffix(".pdf")
    if not arxiv_id:
        raise ValueError(f"Could not extract an arXiv ID from {url!r}")
    return arxiv_id


def download_pdf(
    url: str,
    output_dir: Path,
    *,
    overwrite: bool = False,
    retries: int = 3,
    timeout: float = 60,
    opener: Callable[..., object] = urlopen,
    sleep: Callable[[float], None] = time.sleep,
) -> Path:
    """Download one PDF atomically and return its local path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / f"{arxiv_id_from_url(url)}.pdf"

    if destination.exists() and not overwrite:
        print(f"Skipping existing file: {destination}")
        return destination

    temporary_path = destination.with_suffix(".pdf.part")
    request = Request(url, headers={"User-Agent": USER_AGENT})

    for attempt in range(1, retries + 1):
        try:
            with opener(request, timeout=timeout) as response:
                with temporary_path.open("wb") as output_file:
                    first_chunk = response.read(CHUNK_SIZE)
                    if not first_chunk.startswith(b"%PDF-"):
                        raise ValueError(f"Response from {url} is not a PDF")

                    output_file.write(first_chunk)
                    while chunk := response.read(CHUNK_SIZE):
                        output_file.write(chunk)

            temporary_path.replace(destination)
            print(f"Downloaded: {destination}")
            return destination
        except HTTPError as error:
            if error.code not in RETRYABLE_STATUS_CODES or attempt == retries:
                temporary_path.unlink(missing_ok=True)
                raise
        except URLError:
            if attempt == retries:
                temporary_path.unlink(missing_ok=True)
                raise
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise

        delay = 2 ** (attempt - 1)
        print(f"Download failed; retrying {url} in {delay} second(s)")
        sleep(delay)

    raise RuntimeError(f"Failed to download {url}")


def download_papers(
    urls: Iterable[str],
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    *,
    overwrite: bool = False,
) -> list[Path]:
    """Download all requested papers."""
    return [
        download_pdf(url, output_dir, overwrite=overwrite)
        for url in urls
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download the arXiv PDFs used in the RAG tutorial."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Destination directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace PDFs that have already been downloaded.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = download_papers(
        ARXIV_PDF_URLS,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
    )
    print(f"Ready: {len(paths)} PDF(s) in {args.output_dir}")


if __name__ == "__main__":
    main()
