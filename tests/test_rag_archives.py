"""Verify first-use archive restoration and rejection of damaged or unsafe data.

Small local archives exercise the production extractor without network access.

AI attribution: Modified with AI assistance.
Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

import hashlib
import io
import json
import tarfile

import pytest

from agent_runtime.harness import rag_archives


# Build a controlled archive with real tar metadata so extraction
# tests exercise the same file format and path-safety checks as production.
def bundle(tmp_path, monkeypatch, name, files):
    # files maps archive member paths to bytes; name selects the manifest key
    # requested by the restore call. Splitting this tiny tar stream exercises
    # ordered part assembly and checksums without packaging a live Chroma DB.
    archive = tmp_path / "bundle.tar.xz"
    with tarfile.open(archive, "w:xz") as output:
        for path, content in files.items():
            info = tarfile.TarInfo(path)
            info.size = len(content)
            output.addfile(info, io.BytesIO(content))
    data = archive.read_bytes()
    parts = []
    for number, chunk in enumerate((data[:30], data[30:])):
        filename = f"part{number}"
        (tmp_path / filename).write_bytes(chunk)
        parts.append(
            {
                "file": filename,
                "bytes": len(chunk),
                "sha256": hashlib.sha256(chunk).hexdigest(),
            }
        )
    (tmp_path / "archives.json").write_text(json.dumps({"archives": {name: parts}}))
    monkeypatch.setattr(rag_archives, "ARCHIVE_DIRECTORY", tmp_path)


# The first restore should publish the verified dataset and later
# calls should reuse it without repeating extraction work.
def test_default_restore_and_reuse(tmp_path, monkeypatch):
    bundle(
        tmp_path,
        monkeypatch,
        "index",
        {
            "chroma/db": b"database",
            "config.json": b"{}",
            "manifest.json": b"{}",
        },
    )
    directory = tmp_path / "index"
    monkeypatch.setattr(rag_archives, "DEFAULT_INDEX_DIRECTORY", directory)
    rag_archives.restore_default_index(directory)
    assert (directory / "chroma/db").read_bytes() == b"database"
    (tmp_path / "part0").unlink()
    rag_archives.restore_default_index(directory)  # Existing caches never re-extract.
    with pytest.raises(ValueError, match="overwrite"):
        rag_archives.restore_archive("index", directory)


@pytest.mark.parametrize("damage", ["checksum", "missing", "traversal"])
# Corrupt, missing, and traversal archives must leave no partially
# published index that a later run could mistake for complete data.
def test_failed_restore_does_not_publish(tmp_path, monkeypatch, damage):
    files = {"dataset/shard": b"corpus"}
    if damage == "traversal":
        files["../escaped"] = b"bad"
    bundle(tmp_path, monkeypatch, "dataset", files)
    if damage == "checksum":
        (tmp_path / "part0").write_bytes(b"broken")
    elif damage == "missing":
        (tmp_path / "part0").unlink()
    directory = tmp_path / "index"
    with pytest.raises((ValueError, FileNotFoundError)):
        rag_archives.restore_archive("dataset", directory)
    assert not (directory / "dataset").exists()
    assert not (tmp_path / "escaped").exists()


# Verify the configured bundled source is selected before any network
# download, while still exercising the production builder's next dependency.
def test_build_uses_bundled_dataset(tmp_path, monkeypatch):
    from agent_runtime.harness import rag_index

    bundle(
        tmp_path,
        monkeypatch,
        "dataset",
        {
            f"dataset/wikitext-103-raw-v1/train-{n:05d}-of-00002.parquet": b"fixture"
            for n in range(2)
        },
    )
    downloads = []

    def download(*args, filename, **kwargs):
        # Stop once ingestion requests a tokenizer, proving it skipped corpus download.
        # Record the requested filename and deliberately abort before tokenization or
        # embedding; this test does not build a usable index.
        downloads.append(filename)
        raise RuntimeError("Reached tokenizer download")

    monkeypatch.setattr(rag_index, "hf_hub_download", download)
    with pytest.raises(RuntimeError, match="Reached tokenizer"):
        rag_index.build_index(tmp_path / "index", max_passages=1)
    assert downloads == ["tokenizer.json"]
