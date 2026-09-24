"""Record checksums for complete archives uploaded to GitHub Releases.

Run after package_pip.py and uv build. Release assets live outside Git history;
this manifest lets recipients verify downloads before extracting or installing.
Only the named distributions for the project version are hashed.
AI attribution: Generated with AI assistance by Northstar.
Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

import hashlib
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    """Hash each built artifact; publish the manifest only after all succeed."""
    version = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
    names = (
        "lg-report-pip.tar.gz",
        f"lg_report-{version}.tar.gz",
        f"lg_report-{version}-py3-none-any.whl",
    )
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
