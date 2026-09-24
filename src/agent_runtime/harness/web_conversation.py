"""Record sample turns while the official LangGraph adapter produces AG-UI events.

LangGraph's checkpointer owns browser conversation state; the standard adapter
owns text, tool, state and lifecycle events. This module adds local report
capture and a read-only retained-context preview for the browser.
The web server owns HTTP validation and session registration; this harness
component retains run state and streams events without depending on web routes.
AI attribution: Generated with AI assistance by Northstar.
Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

import asyncio
import logging
from contextlib import aclosing, nullcontext
from datetime import UTC, datetime
from pathlib import Path
from shutil import copyfile
from tempfile import TemporaryDirectory
from time import monotonic

import anyio
from ag_ui.core import CustomEvent, EventType, RunErrorEvent
from langgraph.checkpoint.memory import InMemorySaver
from langsmith import tracing_context

from agent_runtime.harness.context_preview import ContextPreview
from agent_runtime.harness.execution import create_langgraph_agent, open_graph
from agent_runtime.harness.trace_capture import TraceCapture
from agent_runtime.harness.workflow_lifecycle import (
    close_run,
    configure_context_audit,
    save_context_audit,
)
from reporting.pricing import Prices
from reporting.recording import clear_report, save_report

logger = logging.getLogger(__name__)


class WebConversation:
    """Keep one sample's model cursor and native checkpoint across HTTP turns.

    Models with scripted answers must survive between requests just as they do
    in the console loop. A failed/stopped turn requires a new chat because tool
    effects and model cursors cannot safely be rolled back and replayed.
    """

    def __init__(
        self,
        workflow,
        provider,
        model,
        *,
        sample,
        live,
        directory,
        prices=None,
        capture_content=False,
    ):
        """Retain the exact sample factory result without executing the graph."""
        self.workflow = workflow
        self.capture_content = capture_content
        configure_context_audit(workflow, capture_content=capture_content)
        self.provider = provider
        self.model = model
        self.sample = sample
        self.live = live
        self.directory = Path(directory)
        self.checkpointer = InMemorySaver()
        self.context_preview = ContextPreview()
        self.retained_context = None
        self.turn = 0
        self.busy = False
        self.usable = True
        self.pending = []
        self.touched = monotonic()
        self.prices = prices or Prices(
            as_of=datetime.now(UTC).date(),
            note="No price snapshot supplied to chat.",
            models={},
        )

    def close(self):
        """Release recorder and temporary files after this session is inactive."""
        close_run(self.workflow)

    async def run(self, input_data):
        """Yield native events and save reports under the configured capture policy.

        There is no custom token parser, tool callback protocol, socket command
        loop, or cancellation message. Closing the HTTP response cancels the
        native stream; shielding report cleanup preserves partial evidence.
        """
        if not input_data.resume:
            self.turn += 1
        self.usable = False
        self.context_preview.observed_this_turn = False
        # Capture privately while a turn streams. Only the latest completed turn
        # is published, so concurrent chats cannot mix their trace files and no
        # UUID or dated report folders accumulate for the learner.
        workspace = TemporaryDirectory(prefix="lg-report-")
        directory = Path(workspace.name)
        capture = None
        status = "error"
        terminal = None
        try:
            capture = TraceCapture(
                directory / "spans.jsonl",
                self.provider,
                self.model,
                capture_content=self.capture_content,
            )
            async with open_graph(self.workflow) as graph:
                # Serving adds an in-memory checkpoint to the already-built
                # graph. Agent construction, prompts and workflow edges stay in
                # the sample, while the official adapter manages thread history.
                graph.checkpointer = self.checkpointer
                recorder = getattr(graph, "trace_recorder", None)
                agent = create_langgraph_agent(
                    name=self.sample,
                    graph=graph,
                    config={
                        "callbacks": [
                            capture,
                            self.context_preview,
                            *([recorder.callback] if recorder else []),
                        ],
                        "recursion_limit": 30,
                        "metadata": {
                            "report_turn": self.turn,
                            "thread_id": input_data.thread_id,
                        },
                    },
                )
                with (
                    tracing_context(enabled=False),
                    (
                        recorder.scope(
                            input_data.thread_id, input_data.run_id, self.sample
                        )
                        if recorder
                        else nullcontext()
                    ),
                ):
                    async with aclosing(agent.run(input_data)) as events:
                        async for event in events:
                            if event.type == EventType.RUN_ERROR:
                                terminal = RunErrorEvent(
                                    message="The sample failed. Start a new chat; details are in the server log."
                                )
                            elif (
                                event.type == EventType.RUN_FINISHED
                                and terminal is None
                            ):
                                terminal = event
                                outcome = getattr(event, "outcome", None)
                                self.pending = list(getattr(outcome, "interrupts", []))
                                status = (
                                    "interrupted"
                                    if getattr(outcome, "type", None) == "interrupt"
                                    else "ok"
                                )
                            else:
                                yield event
                # Read the final checkpoint, not AG-UI's display transcript or
                # the last model call. edit-with-reloaded-state claims retain a separate working
                # field; its explicit empty list must not fall back to history.
                snapshot = await graph.aget_state(
                    {"configurable": {"thread_id": input_data.thread_id}}
                )
                state = snapshot.values
                history = state.get("working", state.get("messages", []))
                prepare = getattr(graph, "preview_context", None)
                self.retained_context = self.context_preview.describe(
                    history,
                    self.provider,
                    self.model,
                    self.prices,
                    prepared=prepare(state) if prepare else None,
                )
        except asyncio.CancelledError:
            # Let StreamingResponse finish disconnect handling. No fake success
            # event is emitted and no partially executed turn is retried.
            raise
        except Exception:
            logger.exception("Chat run %s failed", input_data.run_id)
            terminal = RunErrorEvent(
                message="Cannot run this sample. Check the server log and start a new chat."
            )
        finally:
            if capture is not None:
                capture.close()
                try:
                    with anyio.CancelScope(shield=True):
                        await asyncio.to_thread(
                            self._save, directory, input_data.run_id, status
                        )
                        output = self.directory
                        clear_report(output)
                        for artifact in directory.iterdir():
                            copyfile(artifact, output / artifact.name)
                except Exception:
                    logger.exception("Cannot save chat report %s", input_data.run_id)
                    status = "error"
                    terminal = RunErrorEvent(
                        message="Could not save the run report. Check the server log."
                    )
            workspace.cleanup()
            self.usable = status in {"ok", "interrupted"}
            self.busy = False
            self.touched = monotonic()
        if self.retained_context is not None:
            yield CustomEvent(name="retained_context", value=self.retained_context)
        if terminal is not None:
            yield terminal

    def _save(self, directory, run_id, status):
        """Save report accounting independently of the next-turn context preview."""
        save_context_audit(self.workflow, directory)
        save_report(
            directory,
            self.prices,
            title=f"{self.sample} · turn {self.turn}",
            demo=not self.live,
            status=status,
            run_id=run_id,
        )
