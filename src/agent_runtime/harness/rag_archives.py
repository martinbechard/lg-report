"""Restore bundled RAG data into the ignored local cache before its first use.

Optional release archives are split into independently downloadable pieces and verified before extraction.
Extract into a temporary directory first; never overwrite an existing index.
The manifest is published last so interrupted restoration cannot appear complete.

AI attribution: Comments updated with AI assistance.

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
    """Make bundled RAG data available locally without downloading it again.

    Return True after installation, or False when no bundle catalog is installed.

    ``name`` must be the catalog key ``index`` or ``dataset`` and ``directory``
    is the destination cache. A missing catalog returns ``False`` so callers can
    use their normal download path; missing/corrupt pieces raise an error instead
    of silently downloading data. Existing payload roots are never overwritten.
    Only regular files and directories in the expected payload are accepted.
    Extraction is staged and moved into place only after integrity and layout
    checks succeed.
    """
    catalog = ARCHIVE_DIRECTORY / "archives.json"
    if not catalog.exists():
        return False
    roots = {
        "index": {"chroma", "config.json", "manifest.json"},
        "dataset": {"dataset"},
    }[name]
    parts = json.loads(catalog.read_text())["archives"][name]
    # Preserve existing cache data before staging. This is a local preflight,
    # not an atomic reservation against another process writing the same paths.
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
                # Verify each split piece before adding it to the reconstructed
                # compressed stream; a corrupt bundle must fail visibly rather
                # than become an apparently valid but different dataset.
                part_path = ARCHIVE_DIRECTORY / filename
                if not part_path.is_file():
                    raise FileNotFoundError(
                        f"Missing RAG asset {filename}. From the repository root, "
                        "run: python scripts/download_rag.py"
                    )
                payload = part_path.read_bytes()
                if (
                    len(payload) != part["bytes"]
                    or hashlib.sha256(payload).hexdigest() != part["sha256"]
                ):
                    raise ValueError(f"Archive integrity check failed: {filename}")
                output.write(payload)
        extracted = staging / "extracted"
        with tarfile.open(archive_path, "r:xz") as archive:
            # Validate every member before extraction: only the expected payload
            # roots and ordinary files/directories belong in this cache. Links
            # and parent/absolute paths could escape that intended destination.
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
        # Sorting puts the completion manifest last: readers must not mistake a
        # partially moved index for a ready one. These moves are not one atomic
        # transaction; failure can leave roots in place for manual diagnosis.
        for root in sorted(roots, key=lambda value: (value == "manifest.json", value)):
            shutil.move(str(extracted / root), directory / root)
    return True


def restore_default_index(directory: Path) -> None:
    """Prepare first-use retrieval from the sample snapshot when the cache is new.

    ``directory`` is the caller's intended index root; return no value. The
    normal build/open operation continues after this optional restore attempt.

    Other directories are deliberately untouched so an explicit cache remains
    under its own download/build policy. Existing files also prevent a restore,
    preserving partial or user-created data for diagnosis.
    """
    if directory.resolve() == DEFAULT_INDEX_DIRECTORY.resolve() and (
        not directory.exists() or not any(directory.iterdir())
    ):
        restore_archive("index", directory)
