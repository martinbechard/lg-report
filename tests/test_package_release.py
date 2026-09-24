"""Verify that release preparation distributes only catalog-listed RAG data.

Tiny local fixtures prove that source/UI bundles are neither required nor
included, and that corrupt archive pieces cannot receive release checksums.
AI attribution: Generated with AI assistance by Northstar.
Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "package_release",
    Path(__file__).resolve().parents[1] / "scripts/package_release.py",
)
package_release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(package_release)


def rag_checkout(tmp_path, monkeypatch):
    """Provide a catalog and one piece without any application build artifacts."""
    monkeypatch.setattr(package_release, "ROOT", tmp_path)
    data = tmp_path / "data/rag"
    data.mkdir(parents=True)
    name = "index.tar.xz.part000"
    payload = b"RAG archive fixture"
    digest = hashlib.sha256(payload).hexdigest()
    (data / name).write_bytes(payload)
    (data / "archives.json").write_text(
        json.dumps(
            {
                "archives": {
                    "index": [{"file": name, "bytes": len(payload), "sha256": digest}]
                }
            }
        )
    )
    return name, payload, digest


def test_release_contains_only_rag(tmp_path, monkeypatch):
    """A fresh destination needs no source bundle, wheel, or frontend build."""
    name, payload, digest = rag_checkout(tmp_path, monkeypatch)
    package_release.main()
    output = tmp_path / "dist"
    assert {path.name for path in output.iterdir()} == {name, "SHA256SUMS"}
    assert (output / name).read_bytes() == payload
    assert (output / "SHA256SUMS").read_text() == f"{digest}  {name}\n"

    # A rerun must not advertise unrelated files already present in dist.
    unrelated = output / "unrelated-build.whl"
    unrelated.write_bytes(b"preserve this build")
    package_release.main()
    assert unrelated.read_bytes() == b"preserve this build"
    assert (output / "SHA256SUMS").read_text() == f"{digest}  {name}\n"


def test_corrupt_rag_has_no_release_manifest(tmp_path, monkeypatch):
    """A same-sized corrupt piece fails verification before it is copied."""
    name, payload, _ = rag_checkout(tmp_path, monkeypatch)
    (tmp_path / "data/rag" / name).write_bytes(b"x" * len(payload))
    with pytest.raises(ValueError, match="RAG integrity check failed"):
        package_release.main()
    assert not (tmp_path / "dist" / name).exists()
    assert not (tmp_path / "dist/SHA256SUMS").exists()
