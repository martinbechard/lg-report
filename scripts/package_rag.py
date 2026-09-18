"""Package a stopped, completed RAG cache as compressed, checksummed Git assets.

Run from the repository root after stopping all ingestion and chat processes.
Only the pinned corpus and completed index are included, never cache credentials.
Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

import hashlib
import json
import pathlib
import tarfile
import tempfile

root = pathlib.Path(".cache/lg-report/rag")
out = pathlib.Path("data/rag")
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
    temporary = tempfile.TemporaryDirectory(prefix="lg-rag-pack-")
    archive = pathlib.Path(temporary.name) / (name + ".tar.xz")

    def normalize(info):
        info.uid = info.gid = 0
        info.uname = info.gname = ""
        info.mtime = 0
        return info

    with tarfile.open(archive, "w:xz", preset=6) as tar:
        for member in members:
            tar.add(root / member, arcname=member, filter=normalize)
    parts = []
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
(out / "archives.json").write_text(json.dumps(manifest, indent=2) + "\n")
