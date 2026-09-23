"""Review native file-tool calls before they can change a local document.

This middleware owns approval interrupts, cancellation, and checkpointed target
observations. The workflow supplies policy and the caller retains persistence;
the file backend separately restricts which local documents tools can access.
No tools are added here. Both synchronous and asynchronous execution use the
same checks, and stale-content checks do not provide atomic write isolation.

AI attribution: Generated with AI assistance by Northstar.
Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

from pathlib import Path
from typing import NotRequired

from deepagents.backends.utils import validate_path
from langchain.agents.middleware import AgentMiddleware, AgentState, hook_config
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.types import Command, interrupt


class ApprovalState(AgentState):
    """Keep execution status and human decisions alongside the agent transcript."""

    target_snapshot: NotRequired[str | None]  # Checkpointed before model proposals.
    status: NotRequired[str]  # Cancellation ends this graph, not just the UI loop.
    changes: NotRequired[list[str]]  # Decisions, not a claim of successful writes.


class RestrictedToolApproval(AgentMiddleware):
    """Intercept restricted calls after the model and before any tool executes.

    This small application middleware uses LangGraph's real interrupt mechanism.
    It spells out approve/reject/cancel for teaching; it is not a prompt-based
    permission check. LangChain also provides HumanInTheLoopMiddleware for its
    standard approval protocol; this sample includes explicit graph cancellation.
    """

    state_schema = ApprovalState

    def __init__(self, restricted_tools, mode, review_context=None):
        """Bind authorization policy independently of the editor implementation."""
        # Fail closed on a typo rather than silently choosing autoapproval.
        if mode not in {"always-ask", "autoapprove"}:
            raise ValueError("Unknown approval mode")
        self.restricted_tools = frozenset(restricted_tools)
        self.mode = mode
        self.review_context = review_context or {}

    def before_model(self, state, runtime):
        """Initialize a baseline when the conversation has no target observation yet.

        This node completes before the approval node can interrupt. Replaying an
        approval therefore cannot replace the baseline with a human's new edit.
        """
        if "target_snapshot" in state:
            return None
        target = Path(self.review_context["target"])
        return {
            "target_snapshot": target.read_text(encoding="utf-8")
            if target.exists()
            else None
        }

    def _check_tool(self, request):
        """Limit this lesson to native file tools and guard approved mutations."""
        call = request.tool_call
        if call["name"] not in {"read_file", *self.restricted_tools}:
            return ToolMessage(
                content="This editor only supports reading, writing, and editing files.",
                tool_call_id=call["id"],
                name=call["name"],
                status="error",
            )
        if call["name"] in self.restricted_tools:
            target = Path(self.review_context["target"])
            current = target.read_text(encoding="utf-8") if target.exists() else None
            # As in the original local sample, comparison and write are not an
            # atomic transaction. This protects pauses, not concurrent writers.
            if current != request.state["target_snapshot"]:
                raise RuntimeError(
                    "Target changed after observation; read and review again"
                )
        return None

    def _read_snapshot(self, request):
        """Capture the baseline alongside an actual target read, not a later turn.

        A checkpointed observation must survive a human edit between the read,
        model proposal, and approval. Reading again deliberately refreshes it.
        """
        if request.tool_call["name"] != "read_file":
            return None
        path = request.tool_call["args"].get("file_path")
        if not isinstance(path, str):
            return None  # Let the native schema report invalid tool arguments.
        try:
            path = validate_path(path)
        except ValueError:
            return None  # The native tool returns its own path-validation error.
        if path == "/target.txt":
            target = Path(self.review_context["target"])
            return {
                "target_snapshot": target.read_text(encoding="utf-8")
                if target.exists()
                else None
            }
        return None

    def wrap_tool_call(self, request, handler):
        """Guard native execution and checkpoint target observations with results."""
        denied = self._check_tool(request)
        if denied is not None:
            return denied
        snapshot = self._read_snapshot(request)
        result = handler(request)
        return (
            Command(update={**snapshot, "messages": [result]})
            if snapshot is not None
            else result
        )

    async def awrap_tool_call(self, request, handler):
        """Apply the identical policy for the browser's asynchronous driver."""
        denied = self._check_tool(request)
        if denied is not None:
            return denied
        snapshot = self._read_snapshot(request)
        result = await handler(request)
        return (
            Command(update={**snapshot, "messages": [result]})
            if snapshot is not None
            else result
        )

    @hook_config(can_jump_to=["end"])
    def after_model(self, state, runtime):
        """Review actual model tool calls, then release, reject, or cancel them."""
        message = state["messages"][-1]
        if not isinstance(message, AIMessage) or not message.tool_calls:
            return {"status": "completed"}
        decisions = list(state.get("changes", []))
        rejected = []
        for call in message.tool_calls:
            if call["name"] not in self.restricted_tools:
                continue
            decision = "approve"
            if self.mode == "always-ask":
                # The model has already selected the tool and exact arguments.
                # Pause here, before the tools node. This node replays on resume;
                # the completed model call is checkpointed and does not replay.
                # For a batch, each interrupt is visited in the same call order.
                decision = interrupt(
                    {
                        "kind": "approval",
                        **self.review_context,
                        "tool": call["name"],
                        "arguments": call["args"],
                        "expected_content": state["target_snapshot"],
                        "choices": ["approve", "reject", "cancel"],
                    }
                )
                # Unknown input creates another interrupt, never authorization.
                # On replay LangGraph supplies earlier answers in order before
                # reaching this new pause; there have still been no tool writes.
                while decision not in ("approve", "reject", "cancel"):
                    decision = interrupt(
                        {
                            "kind": "approval",
                            **self.review_context,
                            "tool": call["name"],
                            "arguments": call["args"],
                            "expected_content": state["target_snapshot"],
                            "error": "Choose approve, reject, or cancel.",
                        }
                    )
            decisions.append(decision)
            if decision == "cancel":
                # End the graph itself. Resolve ALL pending calls as cancelled,
                # including earlier approvals in this unexecuted batch. Keeping
                # call/result pairs avoids leaving an invalid model transcript.
                return {
                    "messages": [
                        ToolMessage(
                            content="Run cancelled; tool not executed.",
                            tool_call_id=item["id"],
                            name=item["name"],
                            status="error",
                        )
                        for item in message.tool_calls
                    ],
                    "changes": decisions,
                    "status": "cancelled",
                    "jump_to": "end",
                }
            if decision == "reject":
                # A result for this call tells create_agent it is resolved, so
                # its tools node skips execution. The agent then sees rejection
                # and decides what to do next; the workflow does not choose edits.
                rejected.append(
                    ToolMessage(
                        content="Human rejected this call. It was not executed. Do not retry it.",
                        tool_call_id=call["id"],
                        name=call["name"],
                        status="error",
                    )
                )
        return {"messages": rejected, "changes": decisions, "status": "running"}
