"""Resume real LangGraph interrupts through a console or scripted human client.

This adapter owns only the approval/resume boundary: the supplied graph still
owns workflow state, while the responder supplies one answer per interrupt.
One recorder spans all pauses; a fresh thread prevents state leaking between
runs. The adapter itself makes no direct model call and invents no token usage;
the supplied graph may still call its configured model while resuming.

Architecture and ownership: docs/chat-composition.md.

AI attribution: Generated with AI assistance.
Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

import json
from uuid import uuid4

from langchain_core.messages import AIMessage
from langgraph.types import Command


class HumanLoop:
    """Adapt an interrupting workflow to the shared recorder's invoke contract.

    ``graph`` must be a compiled LangGraph workflow that uses interrupts and
    accepts ``Command(resume=...)``. ``respond`` receives each interrupt value
    and returns the raw answer for that interrupt. The adapter creates a
    resume mapping keyed by each interrupt ID. It resumes every pending item in
    one batch and returns a recorder-compatible result containing a printable
    final message. Graph and responder exceptions propagate; cancellation is a
    responder policy rather than an invented approval.
    """

    def __init__(self, graph, respond):
        """Prepare a workflow to ask a human whenever execution needs an answer.

        Construction is side-effect free. A new ``HumanLoop`` invocation gets
        its own thread ID, so state from a prior run cannot be resumed by
        accident; callers that need a continuation must keep that responsibility
        in the graph/client layer instead of reusing this adapter's hidden state.
        """
        self.graph = graph
        self.respond = respond

    def invoke(self, inputs, config):
        """Carry one request through human questions or approvals to a final result.

        ``inputs`` is the workflow-specific initial state. The recorder calls
        this method once; this adapter performs all subsequent graph invocations.

        ``config`` is copied and augmented with a generated ``thread_id`` while
        preserving other configurable values and callbacks. The responder is
        called once for every pending interrupt, then the graph is resumed with
        the resulting ``{interrupt_id: answer}`` mapping. The returned mapping
        preserves non-message graph state and replaces the recorder-facing
        message list with a JSON-rendered ``AIMessage``. It prints the final
        JSON result and expects the graph result to be JSON serializable; it
        does not claim provider usage or make a direct model call.
        """
        # A checkpointer uses thread_id to find this run on every later call.
        # Keep this generated ID for the entire loop, but isolate separate runs.
        # Persistence lifetime belongs to the graph's configured checkpointer.
        config = {
            **config,
            "configurable": {
                **config.get("configurable", {}),
                "thread_id": uuid4().hex,
            },
        }
        # invoke runs graph nodes now. If a node calls interrupt(), LangGraph
        # handles its GraphInterrupt exit and returns state with __interrupt__
        # entries; these are questions/approval payloads, not tool results.
        result = self.graph.invoke(inputs, config=config)
        while result.get("__interrupt__"):
            pending = result["__interrupt__"]
            # The responder runs here, outside the interrupted node. Associate
            # each answer with its interrupt ID so multiple pending questions
            # cannot accidentally receive one another's answers.
            answers = {item.id: self.respond(item.value) for item in pending}
            # Command is LangGraph input data; constructing it executes nothing.
            # This invoke restarts each interrupted node from its beginning. Its
            # interrupt call returns the supplied answer on replay, after which
            # that node continues. No suspended Python stack is restored. Earlier
            # completed nodes rerun only if graph edges lead back to them.
            result = self.graph.invoke(Command(resume=answers), config=config)
        print(json.dumps(result, indent=2))
        # AIMessage is LangChain's assistant-message container. This one is
        # authored locally for the recorder; it is neither a model generation
        # nor an executed tool's ToolMessage. The overall return remains a state
        # mapping, with any workflow messages replaced only in this returned copy.
        return {**result, "messages": [AIMessage(content=json.dumps(result, indent=2))]}


def console_response(payload):
    """Let the human review a pending action or question and supply its answer.

    Approval payloads accept the existing approval prompt; other payloads use a
    free-form answer prompt. EOF and Ctrl-C become the workflow's cancellation
    sentinel, allowing an interrupted run to end explicitly instead of hanging
    or being interpreted as approval. The payload is printed in full so a human
    can review the proposed write before answering.
    """
    print(json.dumps(payload, indent=2))
    try:
        return input(
            "approve / reject / cancel: "
            if payload["kind"] == "approval"
            else "Answer (/cancel to abandon): "
        )
    except (EOFError, KeyboardInterrupt):
        return "cancel" if payload["kind"] == "approval" else "/cancel"
