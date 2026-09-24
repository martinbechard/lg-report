"""Download the optional RAG snapshot matching this checkout's checksum catalog.

Uses only Python's standard library, so it also works before pip installation.
Verified pieces are reusable across retries; a partial or corrupt download never
replaces a destination. This downloads archives, not embedding models or packages.
AI attribution: Generated with AI assistance by Northstar.
Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

import argparse
import hashlib
import json
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RELEASES = "https://github.com/martinbechard/lg-report/releases"


def matches(path: Path, part: dict) -> bool:
    """Check both length and digest before reusing or publishing a piece."""
    if not path.is_file() or path.stat().st_size != part["bytes"]:
        return False
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest() == part["sha256"]


def download(directory: Path, base_url: str) -> None:
    """Fetch catalog pieces individually and retain completed work on failure.

    The checked-in catalog is the integrity authority, including when a release
    URL is redirected. An older checkout may need its corresponding release tag.
    Temporary files share the destination filesystem for atomic replacement.
    """
    catalog = json.loads((directory / "archives.json").read_text())
    for parts in catalog["archives"].values():
        for part in parts:
            name = part["file"]
            if Path(name).name != name or name in {".", ".."}:
                raise ValueError(f"Invalid RAG filename: {name}")
            destination = directory / name
            if matches(destination, part):
                print(f"Verified existing {name}", flush=True)
                continue
            print(f"Downloading {name}", flush=True)
            with tempfile.TemporaryDirectory(prefix=".download-", dir=directory) as tmp:
                staged = Path(tmp) / name
                url = f"{base_url}/{urllib.parse.quote(name)}"
                with (
                    urllib.request.urlopen(url, timeout=60) as response,
                    staged.open("wb") as output,
                ):
                    while chunk := response.read(1024 * 1024):
                        output.write(chunk)
                if not matches(staged, part):
                    raise ValueError(f"RAG integrity check failed: {name}")
                staged.replace(destination)
    print("RAG archives verified. The RAG samples restore them on first use.")


def main() -> None:
    """Select the latest release by default, with a tag override for older clones."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tag", help="Release tag containing this checkout's RAG snapshot"
    )
    args = parser.parse_args()
    suffix = (
        f"download/{urllib.parse.quote(args.tag, safe='')}"
        if args.tag
        else "latest/download"
    )
    download(ROOT / "data/rag", f"{RELEASES}/{suffix}")


if __name__ == "__main__":
    main()
