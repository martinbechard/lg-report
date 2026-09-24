"""Build source and compiled Angular for pip-only hosts; RAG is downloaded separately.

Run on the maintainer's machine with Git, Node.js, and npm available. The bundle
preserves repository-relative sample assets, so recipients must use an editable
pip installation from the extracted directory. No Node.js is needed there.
Only source and reference inputs are selected; local credentials, caches, and
generated conversation reports must never travel with the application.
AI attribution: Generated with AI assistance by Northstar.
Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

import subprocess
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIRECTORIES = {"src", "samples", "scripts", "data", "docs", "frontend"}
ROOT_FILES = {
    "README.md",
    "PIP-INSTALL.md",
    "pyproject.toml",
    "LICENSE",
    "NOTICE",
    ".env.example",
    "models.json",
    "models.json.license",
    "exchange-rate.json",
    "exchange-rate.json.license",
}


def source_files():
    """Select current source files, including uncommitted fixes, using Git ignores.

    Explicit path selection excludes reports and runtime state even when tracked.
    Environment templates are allowed, but real environment files are excluded
    independently of Git ignores. Symlinks are omitted rather than shipping data
    outside this checkout. Git is used only on the packaging machine.
    """
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    for name in sorted(set(result.stdout.decode().split("\0")) - {""}):
        relative = Path(name)
        if relative.parts[0] not in SOURCE_DIRECTORIES and name not in ROOT_FILES:
            continue
        if relative.parent == Path("data/rag") and ".tar.xz.part" in relative.name:
            continue
        if any(
            part
            in {"__pycache__", "node_modules", ".angular", ".cache", "reports", "dist"}
            or (part.startswith(".env") and part != ".env.example")
            for part in relative.parts
        ):
            continue
        path = ROOT / relative
        if path.is_file() and not path.is_symlink():
            yield path


def main():
    """Compile a fresh UI before publishing a complete, replaceable archive.

    npm ci uses the frontend lockfile. A failed install/build leaves an existing
    distribution untouched. A temporary archive prevents interrupted packaging
    from replacing the last complete bundle; tar preserves shell script modes.
    """
    for arguments in (["ci"], ["run", "build"]):
        subprocess.run(
            ["npm", "--prefix", "frontend", *arguments], cwd=ROOT, check=True
        )
    frontend = ROOT / "frontend/dist/chat"
    if not (frontend / "browser/index.html").is_file():
        raise RuntimeError("Angular build did not produce browser/index.html")
    destination = ROOT / "dist/lg-report-pip.tar.gz"
    destination.parent.mkdir(exist_ok=True)
    temporary = destination.with_suffix(".tmp")
    try:
        with tarfile.open(temporary, "w:gz", compresslevel=1) as archive:
            # Include Angular's license file alongside the browser assets.
            files = list(source_files()) + sorted(frontend.rglob("*"))
            for path in files:
                if path.is_file() and not path.is_symlink():
                    archive.add(
                        path,
                        arcname=Path("lg-report") / path.relative_to(ROOT),
                        recursive=False,
                    )
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Pip installation bundle: {destination}")


if __name__ == "__main__":
    main()
