"""Prepare only RAG archive pieces and checksums for GitHub Releases.

Run after packaging or downloading the catalog's RAG pieces. Application source
and compiled UI come from the repository, not this release asset collection.
The checksum manifest lets recipients verify the optional RAG download.
AI attribution: Generated with AI assistance by Northstar.
Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    """Copy catalog-listed RAG pieces and record only their release checksums.

    An existing dist directory may contain unrelated builds. Leave those files
    untouched and exclude them from the manifest; no Python or UI build runs.
    """
    names = []
    (ROOT / "dist").mkdir(exist_ok=True)
    # Publish precisely the pieces referenced by the checkout's pinned catalog.
    # Verify before copying so a stale local cache cannot become a bad release.
    catalog = json.loads((ROOT / "data/rag/archives.json").read_text())
    for parts in catalog["archives"].values():
        for part in parts:
            source = ROOT / "data/rag" / part["file"]
            with source.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if source.stat().st_size != part["bytes"] or digest != part["sha256"]:
                raise ValueError(f"RAG integrity check failed: {source.name}")
            shutil.copyfile(source, ROOT / "dist" / source.name)
            names.append(source.name)
    checksums = []
    for name in names:
        # Stream the large RAG bundle instead of retaining it all in memory.
        with (ROOT / "dist" / name).open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        checksums.append(f"{digest}  {name}\n")
    destination = ROOT / "dist" / "SHA256SUMS"
    destination.write_text("".join(checksums))
    print(f"Release checksums: {destination}")


if __name__ == "__main__":
    main()
