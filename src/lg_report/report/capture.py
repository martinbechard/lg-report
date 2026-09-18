"""Translate LangChain lifecycle callbacks into a local OpenTelemetry trace.

AI attribution: Generated with AI assistance.

Capture records evidence, not costs: normalization and pricing remain separate
so reports can be regenerated from saved spans. Messages and tool definitions
are opt-in content; SDK
metadata and error types still explain execution when content capture is off.

Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

import json
from pathlib import Path
from threading import RLock

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.tools import ToolException
from langgraph.errors import GraphInterrupt
from opentelemetry import trace
from opentelemetry.context import Context
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    SimpleSpanProcessor,
    SpanExporter,
    SpanExportResult,
)
from opentelemetry.trace import Status, StatusCode

from lg_report.platform.demo_meter import message_record
from lg_report.report.annotations import describe


class JsonlExporter(SpanExporter):
    """Persist completed spans immediately, without a hosted telemetry service.

    JsonlExporter(Path("spans.jsonl")) requires a new file. Completion-order JSONL
    preserves evidence if a later operation fails; normalize restores start order.
    A lock protects file writes when graph callbacks arrive from worker threads.
    """

    def __init__(self, path: Path):
        """Open a fresh UTF-8 trace file; existing files and I/O failures raise."""
        self.file = path.open("x", encoding="utf-8")
        self.lock = RLock()

    def export(self, spans):
        """Flush completed SDK spans to disk; write errors propagate to the caller."""
        with self.lock:
            for span in spans:
                self.file.write(span.to_json(indent=None) + "\n")
            self.file.flush()
        return SpanExportResult.SUCCESS

    def shutdown(self):
        """Release the file when its owning tracer provider shuts down."""
        with self.lock:
            self.file.close()


class TraceCapture(BaseCallbackHandler):
    """Record one invocation, including nested tools and subagents, as local spans.

    Attach TraceCapture(path, provider, model) to config["callbacks"] and always
    call close(), including after failure. provider/model are identity defaults;
    model callbacks can supply their actual identity. capture_content=True also
    saves prompts, responses, tool definitions and payloads, which may contain private data.

    This owns a private tracer provider rather than changing global OTel state.
    Explicit callback parent IDs preserve ancestry across nested/worker execution;
    a thread's ambient current span is not a reliable graph parent.
    """

    # A broken recorder must fail visibly; silent callback failures could produce
    # a convincing report with missing calls and understated costs.
    raise_error = True

    def __init__(self, path: Path, provider: str, model: str, capture_content=False):
        """Create a local exporter and isolated span registry for this invocation."""
        self.capture_content = capture_content
        self.provider_name = provider
        self.model_name = model
        self.provider = TracerProvider()
        self.provider.add_span_processor(SimpleSpanProcessor(JsonlExporter(path)))
        self.tracer = self.provider.get_tracer("lg-report", "0.1.0")
        self.spans = {}
        self.active = set()
        self.contexts = {}
        self.lock = RLock()

    def _start(self, kind, serialized, run_id, parent_run_id, **kwargs):
        """Open one SDK operation using callback ancestry, not thread ancestry.

        run_id is LangChain's identifier; the OTel span has its own identity.
        Keep the callback ID as an attribute so normalization can reconstruct
        the graph even when a tool or subagent runs on another worker thread.
        """
        with self.lock:
            parent = self.spans.get(parent_run_id)
            # Use the callback parent only when it was captured by this recorder.
            # Otherwise start a root in an empty OTel context, avoiding accidental
            # attachment to unrelated ambient tracing activity.
            context = (
                trace.set_span_in_context(parent, Context()) if parent else Context()
            )
            # Prefer the runtime-assigned name, then the serialized definition;
            # kind is the last usable label when neither supplies a name.
            name = kwargs.get("name") or (serialized or {}).get("name") or kind
            attributes = {"lg.kind": kind, "lg.run_id": str(run_id)}
            metadata = kwargs.get("metadata") or {}
            # Whitelist report metadata rather than copying arbitrary config,
            # which can contain application secrets unrelated to the trace.
            context_info = {}
            for key in (
                "langgraph_node",
                "langgraph_step",
                "report_description",
                "report_purpose",
                "report_turn",
                "cache_ttl",
            ):
                value = metadata.get(key)
                # These fields have scalar meanings in the report schema; omit
                # absent/structured values rather than persist arbitrary objects.
                if isinstance(value, (str, int)):
                    context_info[key] = value
            # Tags help identify graph paths only when supplied; empty/absent
            # tags add no evidence and need no serialized placeholder.
            if kwargs.get("tags"):
                context_info["tags"] = [str(tag) for tag in kwargs["tags"]]
            # Serialized tools may supply a description even without metadata.
            # Only nonempty text is useful; describe below chooses final precedence.
            if (serialized or {}).get("description"):
                context_info["description"] = str(serialized["description"])
            context_info["description"] = describe(
                kind, name, metadata, serialized or {}
            )
            self.contexts[run_id] = context_info
            attributes["lg.context"] = json.dumps(context_info)
            # Preserve the declared callback ancestry even if its span was not
            # found locally; normalization can then expose a broken reference.
            if parent_run_id:
                attributes["lg.parent_run_id"] = str(parent_run_id)
            # Provider identity and reasoning effort describe LLM requests only;
            # attaching them to orchestration/tools would imply extra model calls.
            if kind == "model":
                # Callback identity is more specific than recorder defaults;
                # absent/empty SDK metadata uses the configured provider/model.
                metadata = kwargs.get("metadata") or {}
                attributes["gen_ai.provider.name"] = (
                    metadata.get("ls_provider") or self.provider_name
                )
                invocation_parameters = kwargs.get("invocation_params") or {}
                # Providers expose effort through different request fields. Use
                # the first nonempty explicit value; application metadata is last
                # so it cannot override actual invocation parameters.
                effort = (
                    invocation_parameters.get("reasoning_effort")
                    or (invocation_parameters.get("reasoning") or {}).get("effort")
                    or (invocation_parameters.get("output_config") or {}).get("effort")
                    or metadata.get("report_effort")
                )
                # No effort value means the provider/configuration did not report
                # one; omit it rather than invent a default label.
                if effort:
                    attributes["lg.effort"] = str(effort)
                attributes["gen_ai.request.model"] = (
                    metadata.get("ls_model_name") or self.model_name
                )
            self.spans[run_id] = self.tracer.start_span(
                name, context=context, attributes=attributes
            )
            self.active.add(run_id)
            # Tool schemas are model input, so they share the content opt-in.
            # Capture only supplied definitions, never arbitrary SDK settings.
            if kind == "model" and self.capture_content:
                parameters = kwargs.get("invocation_params") or {}
                # Offline fixtures expose schemas under their own binding key.
                definitions = parameters.get(
                    "tools", parameters.get("tool_definitions")
                )
                if definitions is not None:
                    self._payload(run_id, "tool_definitions", definitions)

    def _end(self, run_id, error=None, usage=None, model=None, interrupted=False):
        """Finish an active callback operation without estimating missing usage.

        usage is the SDK record for this call, model is its returned identifier,
        and interrupted marks a recoverable approval pause rather than failure.
        Repeated completion notifications are harmless because active is checked.
        """
        with self.lock:
            # End callbacks can be repeated or arrive without a captured start.
            # Only active spans may be ended; otherwise we would export twice or
            # access a nonexistent span.
            if run_id not in self.active:
                return
            span = self.spans[run_id]
            # None means usage was not reported. An explicit record, including
            # zero counts, is evidence and must be retained for pricing.
            if usage is not None:
                span.set_attribute("lg.usage", json.dumps(usage))
                span.set_attribute("gen_ai.usage.input_tokens", usage["input_tokens"])
                span.set_attribute("gen_ai.usage.output_tokens", usage["output_tokens"])
            # A returned model identifier can differ from the requested alias;
            # record it when present, otherwise leave request identity available.
            if model:
                span.set_attribute("gen_ai.response.model", model)
            # Either the chain-end marker or GraphInterrupt signals an approval
            # pause: expected control flow, even when delivered as an exception.
            # Only other non-None errors are failures; absence of both means OK.
            if interrupted or isinstance(error, GraphInterrupt):
                span.set_attribute("lg.interrupted", True)
            elif error is not None:
                # Exception text may contain prompts, credentials, or tool content.
                error_type = type(error).__name__
                span.set_attribute("lg.error", error_type)
                span.set_status(Status(StatusCode.ERROR, error_type))
            else:
                span.set_status(Status(StatusCode.OK))
            span.end()
            self.active.remove(run_id)

    def on_chain_start(
        self, serialized, inputs, *, run_id, parent_run_id=None, **kwargs
    ):
        """Track workflow structure even when it has no directly billable usage."""
        self._start("workflow", serialized, run_id, parent_run_id, **kwargs)

    def on_chat_model_start(
        self, serialized, messages, *, run_id, parent_run_id=None, **kwargs
    ):
        """Capture one model request; content is opt-in but role counts remain available."""
        self._start("model", serialized, run_id, parent_run_id, **kwargs)
        # Prompts may contain private data; save them only when content capture
        # was explicitly enabled. Role/count metadata below remains available.
        if self.capture_content:
            self._payload(
                run_id,
                "request",
                [self._message(m) for batch in messages for m in batch],
            )
        self._annotate(
            run_id,
            {
                "message_count": sum(len(batch) for batch in messages),
                "message_roles": [m.type for batch in messages for m in batch],
            },
        )

    _message = staticmethod(message_record)

    def _payload(self, run_id, field, value):
        """Serialize content after the caller has enforced capture permission.

        OTel attributes cannot store nested message dictionaries directly. JSON
        preserves their shape for normalize instead of flattening away tool IDs.
        """
        with self.lock:
            self.spans[run_id].set_attribute(
                f"lg.{field}", json.dumps(value, default=str)
            )

    def _annotate(self, run_id, values):
        """Merge late response metadata with the context saved at operation start.

        The complete JSON attribute is replaced because OTel has no nested merge;
        replacing it with values alone would lose turn and purpose annotations.
        """
        with self.lock:
            self.contexts[run_id].update(values)
            self.spans[run_id].set_attribute(
                "lg.context", json.dumps(self.contexts[run_id])
            )

    def on_llm_start(
        self, serialized, prompts, *, run_id, parent_run_id=None, **kwargs
    ):
        """Support non-chat model lifecycle spans without inventing message records."""
        self._start("model", serialized, run_id, parent_run_id, **kwargs)

    def on_tool_start(
        self, serialized, input_str, *, run_id, parent_run_id=None, **kwargs
    ):
        """Keep tool execution distinct from the model output that requested it."""
        self._start("tool", serialized, run_id, parent_run_id, **kwargs)
        # Tool arguments can expose the same private information as prompts;
        # the content opt-in applies to them as well as model messages.
        if self.capture_content:
            self._payload(
                run_id, "request", [{"role": "tool_call", "content": input_str}]
            )

    def on_retriever_start(
        self, serialized, query, *, run_id, parent_run_id=None, **kwargs
    ):
        """Preserve retrieval timing and ancestry without saving document content."""
        self._start("retriever", serialized, run_id, parent_run_id, **kwargs)

    def on_llm_end(self, response, *, run_id, **kwargs):
        # A chat callback represents one request. Never sum parent graph state messages.
        """Record this request's SDK usage once; missing usage stays unknown."""
        messages = [
            getattr(g, "message", None) for row in response.generations for g in row
        ]
        # Save actual chat messages only with content permission. Non-chat
        # generations have no message object, so the comprehension skips them.
        if self.capture_content:
            self._payload(run_id, "response", [self._message(m) for m in messages if m])
        # Select the first supplied usage record; missing/empty records cannot
        # establish billing. Do not sum candidate generations or graph history.
        usage = next(
            (m.usage_metadata for m in messages if getattr(m, "usage_metadata", None)),
            None,
        )
        # Only message-bearing generations with metadata can identify the actual
        # provider model; absent evidence leaves identity resolution to _start.
        model = next(
            (
                m.response_metadata.get("model_name")
                or m.response_metadata.get("model")
                for m in messages
                if m and m.response_metadata
            ),
            None,
        )
        response_context = {}
        for message in messages:
            # Non-chat generations contain no message metadata to inspect.
            if message:
                response_context.update(message.response_metadata.get("demo_meter", {}))
                # Thinking is content too: require permission AND actual text.
                # A reasoning-token count alone cannot supply text for display.
                if self.capture_content and message.response_metadata.get(
                    "thinking_text"
                ):
                    response_context["thinking_text"] = str(
                        message.response_metadata["thinking_text"]
                    )
                # Preserve an explicitly declared measurement basis (for example
                # simulation units); do not guess one from absent metadata.
                if message.response_metadata.get("usage_basis"):
                    response_context["usage_basis"] = message.response_metadata[
                        "usage_basis"
                    ]
        # Only chat messages carry tool requests. Non-chat generations are
        # skipped; a message without tool_calls contributes no requested names.
        requested_tool_names = [
            call["name"] for m in messages if m for call in getattr(m, "tool_calls", [])
        ]
        # Avoid implying a tool-request event when the response requested none.
        if requested_tool_names:
            response_context["requested_tools"] = requested_tool_names
        for m in messages:
            # Completion reasons belong to chat-message metadata, which is absent
            # for the non-chat generations represented by None here.
            if m:
                # SDKs name this field differently; prefer finish_reason and
                # use stop_reason only when the first is absent/empty.
                reason = m.response_metadata.get(
                    "finish_reason"
                ) or m.response_metadata.get("stop_reason")
                # A missing reason is unknown, not evidence of a normal stop.
                if reason:
                    response_context["finish_reason"] = str(reason)
        self._annotate(run_id, response_context)
        self._end(run_id, usage=usage, model=model)

    def on_chain_end(self, outputs, *, run_id, **kwargs):
        """Preserve an explicit graph interruption as a pause rather than a failure."""
        # Only dictionary graph state supports the interrupt key; a nonempty
        # value proves a pause. Other outputs or empty markers mean normal end.
        self._end(
            run_id,
            interrupted=isinstance(outputs, dict)
            and bool(outputs.get("__interrupt__")),
        )

    def on_tool_end(self, output, *, run_id, **kwargs):
        """Record tool observations when enabled and surface handled tool errors."""
        # Tool results are also opt-in content. SDK message objects expose a
        # type and preserve linkage through _message; plain tool return values
        # use a text wrapper instead because they have no message interface.
        if self.capture_content:
            self._payload(
                run_id,
                "response",
                [
                    self._message(output)
                    if hasattr(output, "type")
                    else {"role": "tool", "content": str(output)}
                ],
            )
        # A tool may return an error-marked ToolMessage without raising. Turn
        # that explicit status into failure evidence; missing/non-error status
        # must not be inferred as failure from the result text.
        self._end(
            run_id,
            error=ToolException()
            if getattr(output, "status", None) == "error"
            else None,
        )

    def on_retriever_end(self, documents, *, run_id, **kwargs):
        """Close retrieval without adding model-token charges for returned documents."""
        self._end(run_id)

    def on_chain_error(self, error, *, run_id, **kwargs):
        """Record workflow failure without persisting potentially sensitive error text."""
        self._end(run_id, error=error)

    def on_llm_error(self, error, *, run_id, **kwargs):
        """Record a failed model call without assuming absent usage means free usage."""
        self._end(run_id, error=error)

    def on_tool_error(self, error, *, run_id, **kwargs):
        """Record tool failure separately from any later model recovery attempt."""
        self._end(run_id, error=error)

    def on_retriever_error(self, error, *, run_id, **kwargs):
        """Record retrieval failure while retaining its place in the graph."""
        self._end(run_id, error=error)

    def close(self):
        """Mark still-open operations incomplete, flush their spans, and close I/O.

        Missing end callbacks must not become successful operations just because
        the caller is cleaning up. Call once after graph execution has stopped.
        """
        with self.lock:
            for run_id in list(self.active):
                self.spans[run_id].set_attribute("lg.incomplete", True)
                self.spans[run_id].end()
            self.active.clear()
            self.provider.shutdown()
