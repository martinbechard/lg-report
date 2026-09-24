"""Package a stopped, completed RAG cache as compressed, checksummed release assets.

Run from the repository root after stopping all ingestion and chat processes.
Only the pinned corpus and completed index are included, never cache credentials.
Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

# AI attribution: Generated with AI assistance.
# This operator script performs its packaging work at invocation time so a
# command either produces a complete checksummed artifact set or stops before
# writing archive metadata.

import hashlib
import json
import pathlib
import tarfile
import tempfile

root = pathlib.Path(".cache/lg-report/rag")
out = pathlib.Path("data/rag")
# These repository-relative paths make the invocation reproducible. Run from
# the repository root only after ingestion and chat processes have stopped, so
# the archive cannot race an active index writer.
out.mkdir(parents=True, exist_ok=True)
if not (root / "manifest.json").is_file():
    raise SystemExit("Finish ingestion before packaging the index")
manifest = {
    "copyright": "Copyright (c) 2026 Martin.Bechard@DevConsult.ca; packaging only, third-party content retains its license.",
    "archives": {},
}
for name, members in [
    ("index", ["chroma", "config.json", "manifest.json"]),
    ("dataset", ["dataset/wikitext-103-raw-v1"]),
]:
    # Separate index and dataset archives keep large artifacts independently
    # transferable. Normalized tar metadata below makes checksums stable when
    # the source cache has not changed.
    temporary = tempfile.TemporaryDirectory(prefix="lg-rag-pack-")
    archive = pathlib.Path(temporary.name) / (name + ".tar.xz")

    def normalize(info):
        """Make identical cache contents produce stable archive metadata.

        tar.add calls this filter for each member before storing its header.
        info is the mutable TarInfo for that member. Clear host-specific owner
        names/IDs and modification time, then return the same header; payload
        content and member paths are preserved.
        """
        info.uid = info.gid = 0
        info.uname = info.gname = ""
        info.mtime = 0
        return info

    with tarfile.open(archive, "w:xz", preset=6) as tar:
        for member in members:
            tar.add(root / member, arcname=member, filter=normalize)
    parts = []
    # Split archives into bounded parts for independently retryable release downloads. Each
    # part gets its own digest and license sidecar; archives.json records the
    # complete set needed for reconstruction.
    with archive.open("rb") as stream:
        while chunk := stream.read(48 * 1024 * 1024):
            part = out / f"{name}.tar.xz.part{len(parts):03d}"
            part.write_bytes(chunk)
            pathlib.Path(str(part) + ".license").write_text(
                "Copyright (c) 2026 Martin.Bechard@DevConsult.ca (packaging).\n"
                "Dataset and derived text: CC-BY-SA-3.0; upstream Wikipedia contributors.\n"
                "See README.md and ../../NOTICE for attribution and scope.\n"
            )
            parts.append(
                {
                    "file": part.name,
                    "bytes": len(chunk),
                    "sha256": hashlib.sha256(chunk).hexdigest(),
                }
            )
    manifest["archives"][name] = parts
    print(name, archive.stat().st_size, flush=True)
    temporary.cleanup()
# Publish the reconstruction catalog after both sets of pieces exist. This
# ordering is not an atomic replacement of an earlier archive set: operators
# must avoid packaging concurrently with consumers or an active cache writer.
(out / "archives.json").write_text(json.dumps(manifest, indent=2) + "\n")
