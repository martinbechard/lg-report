"""Record checksums for complete archives uploaded to GitHub Releases.

Run after package_pip.py and uv build. Release assets live outside Git history;
this manifest lets recipients verify downloads before extracting or installing.
Only the named distributions for the project version are hashed.
AI attribution: Generated with AI assistance by Northstar.
Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

import hashlib
import json
import shutil
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    """Hash each built artifact; publish the manifest only after all succeed."""
    version = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
    names = [
        "lg-report-pip.tar.gz",
        f"lg_report-{version}.tar.gz",
        f"lg_report-{version}-py3-none-any.whl",
    ]
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
