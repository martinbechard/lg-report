"""Restore bundled RAG data into the ignored local cache before its first use.

Archives are split into Git-friendly pieces and verified before extraction.
Extract into a temporary directory first; never overwrite an existing index.
The manifest is published last so interrupted restoration cannot appear complete.
Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

import hashlib
import json
import shutil
import tarfile
import tempfile
from pathlib import Path, PurePosixPath

PROJECT_ROOT = Path(__file__).resolve().parents[3]
ARCHIVE_DIRECTORY = PROJECT_ROOT / "data/rag"
DEFAULT_INDEX_DIRECTORY = PROJECT_ROOT / ".cache/lg-report/rag"


def restore_archive(name: str, directory: Path) -> bool:
    """Restore a bundled archive, or return False when no bundle is installed.

    Missing/corrupt pieces raise an error instead of silently downloading data.
    Only regular files and directories in the expected payload are accepted.
    """
    catalog = ARCHIVE_DIRECTORY / "archives.json"
    if not catalog.exists():
        return False
    roots = {
        "index": {"chroma", "config.json", "manifest.json"},
        "dataset": {"dataset"},
    }[name]
    parts = json.loads(catalog.read_text())["archives"][name]
    if any((directory / root).exists() for root in roots):
        raise ValueError(f"Refusing to overwrite existing {name} files in {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".restore-", dir=directory) as temporary:
        staging = Path(temporary)
        archive_path = staging / "archive.tar.xz"
        with archive_path.open("wb") as output:
            for part in parts:
                filename = part["file"]
                if Path(filename).name != filename:
                    raise ValueError("Invalid archive part filename")
                payload = (ARCHIVE_DIRECTORY / filename).read_bytes()
                if (
                    len(payload) != part["bytes"]
                    or hashlib.sha256(payload).hexdigest() != part["sha256"]
                ):
                    raise ValueError(f"Archive integrity check failed: {filename}")
                output.write(payload)
        extracted = staging / "extracted"
        with tarfile.open(archive_path, "r:xz") as archive:
            for member in archive.getmembers():
                path = PurePosixPath(member.name)
                if (
                    path.is_absolute()
                    or ".." in path.parts
                    or not path.parts
                    or path.parts[0] not in roots
                    or not (member.isfile() or member.isdir())
                ):
                    raise ValueError(f"Unsafe archive member: {member.name}")
            archive.extractall(extracted, filter="data")
        if {p.name for p in extracted.iterdir()} != roots:
            raise ValueError(f"Incomplete {name} archive")
        for root in sorted(roots, key=lambda value: (value == "manifest.json", value)):
            shutil.move(str(extracted / root), directory / root)
    return True


def restore_default_index(directory: Path) -> None:
    """Expand the sample snapshot only for a pristine default cache location."""
    if directory.resolve() == DEFAULT_INDEX_DIRECTORY.resolve() and (
        not directory.exists() or not any(directory.iterdir())
    ):
        restore_archive("index", directory)
