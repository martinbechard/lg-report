"""Bind DeepAgent's native file tools to one source and one writable target.

The backend translates two virtual filenames into trusted local paths, then
lets DeepAgent's FilesystemBackend perform reads, writes, and replacements.
Other backend operations remain unsupported; this lesson needs no directory
search, shell, delegation, or arbitrary filesystem access.

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


class FileAccessBackend(BackendProtocol):
    """Expose only the configured documents through standard backend operations."""

    def __init__(self, source, target):
        """Bind trusted paths without opening either document."""
        self.paths = {
            "/source.txt": Path(source).resolve(),
            "/target.txt": Path(target).resolve(),
        }
        # Real paths are chosen here, never by the model. The underlying backend
        # retains the library's UTF-8 handling and string-replacement behavior.
        self.filesystem = FilesystemBackend(virtual_mode=False)

    def _virtual_result(self, result, file_path):
        """Keep successful paths and library error messages in the virtual namespace."""
        if result.error:
            result.error = result.error.replace(str(self.paths[file_path]), file_path)
        if isinstance(result, (WriteResult, EditResult)):
            result.path = file_path if not result.error else None
        return result

    def read(self, file_path: str, offset: int = 0, limit: int = 2000) -> ReadResult:
        """Allow native paginated reads of either configured document."""
        if file_path not in self.paths:
            return ReadResult(error="Only /source.txt and /target.txt may be read")
        result = self.filesystem.read(str(self.paths[file_path]), offset, limit)
        return self._virtual_result(result, file_path)

    def write(self, file_path: str, content: str) -> WriteResult:
        """Delegate full-content writes only for the configured target."""
        if file_path != "/target.txt":
            return WriteResult(error="Only /target.txt may be written")
        result = self.filesystem.write(str(self.paths[file_path]), content)
        return self._virtual_result(result, file_path)

    def edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> EditResult:
        """Delegate native string replacement with the same target restriction."""
        if file_path != "/target.txt":
            return EditResult(error="Only /target.txt may be edited")
        result = self.filesystem.edit(
            str(self.paths[file_path]), old_string, new_string, replace_all
        )
        return self._virtual_result(result, file_path)
