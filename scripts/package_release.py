"""Publish rebuilt distributions as Git-sized parts with integrity checks.

Run after package_pip.py and uv build. Complete archives remain in dist for
delivery; numbered parts let GitHub store the same bytes below its file limit.
Only the three named release artifacts are copied, never arbitrary local files.
AI attribution: Generated with AI assistance by Northstar.
Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = (
    "lg-report-pip.tar.gz",
    "lg_report-0.1.0.tar.gz",
    "lg_report-0.1.0-py3-none-any.whl",
)
PART_SIZE = 45 * 1024 * 1024


def main():
    """Split complete archives and record hashes of both parts and originals."""
    sources = [ROOT / "dist" / name for name in ARTIFACTS]
    # Validate inputs before replacing an earlier release's numbered parts.
    for source in sources:
        if not source.is_file():
            raise FileNotFoundError(source)
    destination = ROOT / "dist" / "release"
    destination.mkdir(exist_ok=True)
    original_hashes = []
    part_hashes = []
    for source in sources:
        digest = hashlib.sha256()
        with source.open("rb") as stream:
            index = 0
            while block := stream.read(PART_SIZE):
                index += 1
                name = f"{source.name}.part{index:03d}"
                (destination / name).write_bytes(block)
                digest.update(block)
                part_hashes.append(f"{hashlib.sha256(block).hexdigest()}  {name}\n")
        # Remove only obsolete parts for this exact artifact when it shrinks.
        for part in destination.glob(f"{source.name}.part[0-9][0-9][0-9]"):
            if int(part.name[-3:]) > index:
                part.unlink()
        original_hashes.append(f"{digest.hexdigest()}  ../{source.name}\n")
    (destination / "PARTS.sha256").write_text("".join(part_hashes))
    (destination / "ARCHIVES.sha256").write_text("".join(original_hashes))
    print(f"Release parts and checksums: {destination}")


if __name__ == "__main__":
    main()
