"""Check optional RAG downloads without a network or the large real corpus.

Local URL fixtures exercise checksum verification, retry reuse, and preservation
of existing files on corrupt downloads, using the production download helper.
AI attribution: Generated with AI assistance by Northstar.
Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "download_rag", Path(__file__).resolve().parents[1] / "scripts/download_rag.py"
)
download_rag = importlib.util.module_from_spec(spec)
spec.loader.exec_module(download_rag)


def setup_catalog(tmp_path):
    """Make one tiny pinned piece and a separate file-URL release directory."""
    destination = tmp_path / "checkout"
    destination.mkdir()
    release = tmp_path / "release"
    release.mkdir()
    payload = b"compressed fixture"
    name = "index.tar.xz.part000"
    (release / name).write_bytes(payload)
    (destination / "archives.json").write_text(
        json.dumps(
            {
                "archives": {
                    "index": [
                        {
                            "file": name,
                            "bytes": len(payload),
                            "sha256": hashlib.sha256(payload).hexdigest(),
                        }
                    ]
                }
            }
        )
    )
    return destination, release, name, payload


def test_download_and_offline_reuse(tmp_path):
    """A verified piece is usable and a retry requires no release connection."""
    destination, release, name, payload = setup_catalog(tmp_path)
    download_rag.download(destination, release.as_uri())
    assert (destination / name).read_bytes() == payload
    (release / name).unlink()
    download_rag.download(destination, release.as_uri())


def test_corrupt_download_preserves_destination(tmp_path):
    """Bad bytes cannot replace a local piece or leave a download staging tree."""
    destination, release, name, payload = setup_catalog(tmp_path)
    (destination / name).write_bytes(b"previous local bytes")
    (release / name).write_bytes(b"x" * len(payload))
    with pytest.raises(ValueError, match="integrity"):
        download_rag.download(destination, release.as_uri())
    assert (destination / name).read_bytes() == b"previous local bytes"
    assert not list(destination.glob(".download-*"))
