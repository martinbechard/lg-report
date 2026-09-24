"""Restrict native file tools to the shared teaching exercise workspace.

The context-budget and circuit-breaker workflows share these virtual paths. Models
cannot select another local path; the reviewer's middleware omits write tools.

AI attribution: Generated with AI assistance by Northstar.
Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

from pathlib import Path

from deepagents.backends import FilesystemBackend
from deepagents.backends.protocol import (
    BackendProtocol,
    EditResult,
    ReadResult,
    WriteResult,
)


class ExerciseBackend(BackendProtocol):
    """Map exactly three virtual documents into one caller-owned directory."""

    def __init__(self, directory: str | Path, *, writable_paths=None):
        root = Path(directory).resolve()
        root.mkdir(parents=True, exist_ok=True)
        self.paths = {f"/{name}": root / name for name in ("plan.md", "slug.py", "test_slug.py")}
        # Role ownership is independent of model instructions and compaction.
        self.writable_paths = set(self.paths if writable_paths is None else writable_paths)
        self.filesystem = FilesystemBackend(virtual_mode=False)

    def _translate(self, result, path):
        """Keep real locations out of model-facing success and error messages."""
        if result.error:
            result.error = result.error.replace(str(self.paths[path]), path)
        if isinstance(result, (WriteResult, EditResult)):
            result.path = path if not result.error else None
        return result

    def read(self, file_path: str, offset: int = 0, limit: int = 2000) -> ReadResult:
        """Read an allowed document, with native pagination for long plans."""
        if file_path not in self.paths:
            return ReadResult(error="Only /plan.md, /slug.py, and /test_slug.py may be read")
        return self._translate(self.filesystem.read(str(self.paths[file_path]), offset, limit), file_path)

    def write(self, file_path: str, content: str) -> WriteResult:
        """Write only an allowed document selected by the workflow."""
        if file_path not in self.paths:
            return WriteResult(error="Only /plan.md, /slug.py, and /test_slug.py may be written")
        if file_path not in self.writable_paths:
            return WriteResult(error="This role cannot write that path")
        return self._translate(self.filesystem.write(str(self.paths[file_path]), content), file_path)

    def edit(self, file_path: str, old_string: str, new_string: str, replace_all: bool = False) -> EditResult:
        """Use the library's exact-match edit semantics within allowed paths."""
        if file_path not in self.paths:
            return EditResult(error="Only /plan.md, /slug.py, and /test_slug.py may be edited")
        if file_path not in self.writable_paths:
            return EditResult(error="This role cannot edit that path")
        return self._translate(
            self.filesystem.edit(str(self.paths[file_path]), old_string, new_string, replace_all),
            file_path,
        )
