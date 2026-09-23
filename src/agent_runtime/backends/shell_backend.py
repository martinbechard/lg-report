"""Provide local shell execution for DeepAgent's native execute tool.

ShellBackend implements SandboxBackendProtocol and reuses FilesystemBackend for
native file tools. The protocol name does not imply isolation: commands run as
the current user, with the current environment, in a configured working directory.
This POSIX teaching backend is intended for trusted scripts. A timeout kills the
command's process group; it cannot undo file changes already made by the script.

AI attribution: Generated with AI assistance by Northstar.
Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

import os
import shutil
import signal
import subprocess
import tempfile
from pathlib import Path
from uuid import uuid4

from deepagents.backends import FilesystemBackend
from deepagents.backends.protocol import ExecuteResponse, SandboxBackendProtocol


class ShellBackend(FilesystemBackend, SandboxBackendProtocol):
    """Combine native file storage with a local POSIX command runner."""

    def __init__(self, working_directory: str | Path):
        """Bind the workspace and shell without executing a command."""
        if os.name != "posix":
            raise RuntimeError("ShellBackend requires a POSIX host with sh")
        self.shell = shutil.which("sh")
        if self.shell is None:
            raise RuntimeError("ShellBackend requires sh on PATH")
        super().__init__(root_dir=working_directory, virtual_mode=True)
        self._id = f"local-shell-{uuid4()}"

    @property
    def id(self) -> str:
        """Identify this backend instance without exposing a temporary host path."""
        return self._id

    def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
        """Run one command and return combined output, status, and truncation.

        None uses a 30-second default; zero disables the deadline. The native
        tool reports nonzero exits to the model rather than claiming success.
        The protocol's aexecute implementation runs this method in a worker
        thread, so browser execution does not block the event loop.
        """
        if timeout is not None and timeout < 0:
            raise ValueError("timeout must be non-negative")
        deadline = 30 if timeout is None else timeout or None
        # Capture into a temporary file rather than an unbounded memory buffer.
        # The returned observation is capped; this is not a disk quota or sandbox.
        with (
            tempfile.TemporaryFile() as output,
            subprocess.Popen(
                [self.shell, "-c", command],
                cwd=self.cwd,
                stdin=subprocess.DEVNULL,
                stdout=output,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            ) as process,
        ):
            timed_out = False
            try:
                process.wait(timeout=deadline)
            except subprocess.TimeoutExpired:
                timed_out = True
                # Killing only sh can leave its child script running. The
                # new session gives this invocation a separate process group.
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass  # The process group exited as the deadline elapsed.
                process.wait()
            output.seek(0)
            captured = output.read(20_001)
        truncated = len(captured) > 20_000
        text = captured[:20_000].decode("utf-8", errors="replace")
        if timed_out:
            text += f"\nCommand timed out after {deadline} seconds."
        return ExecuteResponse(
            output=text,
            exit_code=124 if timed_out else process.returncode,
            truncated=truncated,
        )
