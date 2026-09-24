"""Contrast retained claim history with a fresh view after a successful edit.

The claim store is authoritative; model messages are snapshots. This deliberately
small, single-claim workflow composes the named claims_agent with a context-policy graph.
The agent owns instructions, tools, and domain state; this module owns the
working context and its audit. See docs/chat-composition.md.
No insurance decisions or disk-file edits are performed. A client supplies turns,
and the application keeps an audit separate from the model's working messages.

AI attribution: Generated with AI assistance (Northstar).
Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

import json
from typing import Literal

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.runnables.config import merge_configs
from langgraph.graph import END, START, MessagesState, StateGraph

from agent_runtime.agents.claims_agent import build_agent
from agent_runtime.harness.demo_meter import message_record
from agent_runtime.harness.model_factory import build_model

Mode = Literal["edit-with-patched-state", "edit-with-reloaded-state"]


class ContextAudit(BaseCallbackHandler):
    """Capture actual model inputs separately from the editable working history.

    Counts are message counts, not token estimates. Content capture and printing
    are explicit options; metadata-only runs retain neither claim nor prompt text
    in this audit. The regular report callback still owns token/cost accounting.
    """

    def __init__(self, *, capture_content=True, show_context=False, write=print):
        """Start a fresh audit; the conversation sets turn before each invocation."""
        # Content suppression applies to the extra teaching audit as well as the
        # standard report. Otherwise metadata-only could still leak record text.
        self.capture_content = capture_content
        self.show_context = show_context and capture_content
        self.write = write
        self.turn = 0
        self.calls = []

    def on_chat_model_start(self, serialized, messages, **kwargs):
        """Snapshot the inputs actually sent, including the agent system prompt."""
        # Observe the actual request, not the harness's intended history: internal
        # tool loops add messages between calls. This exposes what the model saw
        # before a read, after an edit, and after context invalidation.
        for batch in messages:
            entry = {"turn": self.turn, "message_count": len(batch)}
            if self.capture_content:
                entry["messages"] = [message_record(message) for message in batch]
            self.calls.append(entry)
            self.write(
                f"Model call {len(self.calls)} · turn {self.turn} · {len(batch)} messages"
            )
            if self.show_context:
                self.write(json.dumps(entry, indent=2))


class ClaimsState(MessagesState):
    """Keep the client's visible conversation separate from future model input.

    Inherited ``messages`` uses LangGraph's message reducer: new messages are
    added and matching IDs updated. ``working`` has no reducer, so each turn
    replaces it wholesale. That difference lets the UI keep an old claim read
    while the next model request omits it. The driver checkpoints these fields;
    neither field is the authoritative claim, which remains inside the agent.
    """

    # Missing on the first turn means no prior context. Later values contain the
    # agent's completed history or its edit-with-reloaded-state projection, including any notice
    # and retained policy exchanges. Never rebuild this from display messages.
    working: list


class ClaimsContext:
    """Run the same agent with either retained or purged working messages.

    edit-with-reloaded-state mode removes claim exchanges and conversation prose after an edit,
    while preserving actual policy tool exchanges. The agent owns all reads.
    This coarse policy is intentional for one claim: it prevents stale facts from
    surviving indirectly. It also loses earlier preferences and cannot answer
    historical questions. The separate audit remains available to the human.
    """

    def __init__(self, agent, mode: Mode, *, audit=None, write=print):
        """Wrap an agent with a workflow-selected working-context strategy.

        The harness sees only invoke, context_version, retain_unaffected_context,
        and evidence. It never opens the agent's store, names a tool, or supplies
        role instructions. The agent describes what remains valid; this harness
        decides whether and when to use that projection instead of full history.
        """
        # Validate strategy here because it is a harness setting. The agent is
        # deliberately constructed without this flag and behaves identically
        # given identical working messages in either experiment.
        if mode not in {"edit-with-patched-state", "edit-with-reloaded-state"}:
            raise ValueError(
                "mode must be edit-with-patched-state or edit-with-reloaded-state"
            )
        self.mode = mode
        self.agent = agent
        self.audit = audit or ContextAudit(write=write)
        self.write = write
        # Events describe context-policy decisions for reporting. They are not
        # messages and never enter the agent's input or token-accounting stream.
        self.events = []
        # A convenient view of the most recently selected working context. The
        # checkpointed state supplies continuity to turn(); this attribute is not
        # used to reconstruct a missing checkpoint or persist the claim store.
        self.history = []

        # Policy is a graph node, independent of how a client supplies turns.
        # The driver's checkpoint retains the UI transcript and working context
        # separately. Old UI messages must never repopulate purged model context.
        builder = StateGraph(ClaimsState)
        builder.add_node("claim_turn", self.turn)
        builder.add_edge(START, "claim_turn")
        builder.add_edge("claim_turn", END)
        self.graph = builder.compile()

    def turn(self, state, config):
        """Answer the newest client request, then choose the next turn's context.

        The conversation driver supplies checkpointed state plus a new message;
        config carries callbacks and run metadata through the inner agent graph.
        invoke returns an agent state dictionary whose messages include the
        supplied working prefix, the new request, tool exchanges, and the answer.

        Return a partial graph-state update: display messages for this turn and
        a complete replacement for working context. Invalidation occurs after
        invoke returns, so all tool iterations in this turn still see its history.
        Invocation errors propagate; this method does not roll back tool edits
        or apply its normal end-of-turn projection after an exception.
        """
        self.audit.turn += 1
        # Merge rather than replace callbacks so report tracing and this teaching
        # audit observe the same inner model calls, including tool-loop retries.
        turn_config = merge_configs(config, {"callbacks": [self.audit]})
        working = list(state.get("working", []))
        # A changed version proves the store changed, unlike a model's proposed
        # edit or its prose acknowledgement. Failed edits alone keep this value.
        before_version = self.agent.context_version
        # Only the newest display message enters the inner graph. Passing the
        # whole UI transcript would silently restore previously purged snapshots.
        result = self.agent.invoke(
            {"messages": [*working, state["messages"][-1]]}, config=turn_config
        )
        self.history = list(result["messages"])
        # One projection covers all successful edits in a completed turn. edit-with-patched-state
        # mode keeps the full result; edit-with-reloaded-state mode with no change does the same.
        # This requests a message projection, never a record read or model call.
        if (
            self.mode == "edit-with-reloaded-state"
            and self.agent.context_version != before_version
        ):
            previous = len(self.history)
            self.history = self.agent.retain_unaffected_context(self.history)
            self.events.append(
                {
                    "event": "purge",
                    "turn": self.audit.turn,
                    "previous_messages": previous,
                    "retained_tool_results": sum(
                        m.type == "tool" for m in self.history
                    ),
                    "context_version": self.agent.context_version,
                }
            )
            self.write(
                "Applied edit-with-reloaded-state context strategy using the agent retention rules. No automatic reads."
            )
        # Return only this turn's display messages. The working projection is a
        # replacement field, so checkpoint history cannot merge removed snapshots
        # back into the next model input. The agent still chooses every read.
        # Slicing off the supplied working prefix also excludes retained notices
        # and old policy pairs from being emitted to the display as new activity.
        # The new human message may already exist there; its ID lets the reducer
        # reconcile it rather than append a second copy.
        return {"messages": result["messages"][len(working) :], "working": self.history}

    def evidence(self):
        """Return report data describing calls, purges, and optionally final state.

        Calls describe actual model inputs, not merely one entry per user turn;
        an agent may make several model calls while using tools. Metadata-only
        capture retains counts/events but excludes messages and the final claim.
        The calls/events lists are this session's accumulated audit, not a deep
        copy for a caller to mutate. Reporting must treat them as read-only.
        """
        # The audit answers "what happened?" independently of the working context
        # answering "what should the agent see next?" Keep both, but never merge
        # the historical audit back into a subsequent model request.
        result = {"mode": self.mode, "calls": self.audit.calls, "events": self.events}
        if self.audit.capture_content:
            result.update(self.agent.evidence())
        return result


def build_workflow(
    model=None, mode: Mode = "edit-with-reloaded-state", *, audit=None, write=print
):
    """Compose the named claims agent with the workflow's selected context strategy.

    Model configuration and user input are application concerns. Agent internals
    remain encapsulated behind its interface; the harness receives a component,
    not a tool list, prompt, or claim store.
    """
    # Model selection belongs to the factory; the workflow supplies its role.
    if model is None:
        model = build_model(caller="workflow")
    # Composition selects the participant. The conversation harness only needs
    # its public methods, so another encapsulated agent could be wrapped without
    # moving that agent's instructions or tool logic into this workflow.
    return ClaimsContext(build_agent({"model": model}), mode, audit=audit, write=write)
