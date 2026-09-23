"""Compose the file editor with the requested approval policy.

The agent owns prompts and edit choices. RestrictedToolApproval implements the
approval, cancellation, and stale-content checks; FileAccessBackend restricts
local file access. The caller owns the checkpointer across turns and resumes.

AI attribution: Generated with AI assistance by Northstar.
Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

from pathlib import Path

from deepagents.middleware.filesystem import FilesystemMiddleware

from agent_runtime.agents.file_editor import build_agent
from agent_runtime.backends.file_access_backend import FileAccessBackend
from agent_runtime.harness.model_factory import build_model
from agent_runtime.middleware.restricted_tool_approval import RestrictedToolApproval

# Both native mutation tools must pass the same approval and stale-content gate.
RESTRICTED_TOOLS = frozenset({"write_file", "edit_file"})


def build_workflow(
    model=None, *, source=None, target=None, mode="always-ask", checkpointer=None
):
    """Configure editing policy with caller-owned persistence.

    Direct callers supply a checkpointer for interrupt/resume. Conversation
    entry points may omit it here because they attach their retained saver
    before execution. This factory never allocates or owns a saver.
    """
    from tempfile import TemporaryDirectory

    import samples.file_approval

    workspace = (
        TemporaryDirectory(prefix="lg-file-approval-") if target is None else None
    )
    target = Path(workspace.name) / "edited-summary.txt" if workspace else Path(target)
    source = source or Path(samples.file_approval.__file__).with_name("input.txt")
    try:
        if model is None:
            model = build_model(caller="workflow")
        graph = _compose(
            model, source=source, target=target, mode=mode, checkpointer=checkpointer
        )
    except BaseException:
        if workspace:
            workspace.cleanup()
        raise
    graph.workspace = workspace
    graph.output_file = target
    return graph


def _compose(model, *, source, target, mode, checkpointer):
    """Keep approval policy identical for explicit paths and temporary sessions."""
    # The harness supplies policy through the agent's public construction seam.
    # It neither reads the source nor precomputes tool calls from a list of edits.
    # The same policy runs for scripted model fixtures and live model decisions.
    #
    # human request -> agent model -> approval middleware
    #                    ^                  |
    #                    |   unrestricted / autoapprove / approved
    #                    |                  v
    #                    +-------------- tools
    #                    |
    #                    +-- rejected tool result (no tool execution)
    #
    # restricted + always-ask -> interrupt -> human -> resume same checkpoint
    # cancel -> END; model final answer -> END
    #
    # Keep the caller-owned checkpointer and thread_id across resume calls.
    # Persistence durability depends on the supplied saver. A successful earlier
    # write is never undone by rejecting or cancelling a later proposal.
    backend = FileAccessBackend(source, target)
    # Compose the library's native file tools with this workflow's approval
    # policy. No implicit subagent can execute outside that policy boundary.
    parameters = {
        "model": model,
        "middleware": (
            FilesystemMiddleware(
                backend=backend,
                tools=["read_file", "write_file", "edit_file"],
            ),
            RestrictedToolApproval(
                RESTRICTED_TOOLS, mode, {"target": str(Path(target).resolve())}
            ),
        ),
        "checkpointer": checkpointer,
    }
    return build_agent(parameters)
