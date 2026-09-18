"""LangChain callbacks -> local OpenTelemetry spans; no hosted exporter required."""

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

from .annotations import describe
from .demo_meter import message_record


class JsonlExporter(SpanExporter):
    """One OpenTelemetry SDK span JSON object per line, flushed on completion."""

    def __init__(self, path: Path):
        self.file = path.open("x", encoding="utf-8")
        self.lock = RLock()

    def export(self, spans):
        with self.lock:
            for span in spans:
                self.file.write(span.to_json(indent=None) + "\n")
            self.file.flush()
        return SpanExportResult.SUCCESS

    def shutdown(self):
        with self.lock:
            self.file.close()


class TraceCapture(BaseCallbackHandler):
    """One capture per invocation. Explicit parents preserve concurrent graph topology."""

    raise_error = True

    def __init__(self, path: Path, provider: str, model: str, capture_content=False):
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
        with self.lock:
            parent = self.spans.get(parent_run_id)
            context = (
                trace.set_span_in_context(parent, Context()) if parent else Context()
            )
            name = kwargs.get("name") or (serialized or {}).get("name") or kind
            attrs = {"lg.kind": kind, "lg.run_id": str(run_id)}
            metadata = kwargs.get("metadata") or {}
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
                if isinstance(value, (str, int)):
                    context_info[key] = value
            if kwargs.get("tags"):
                context_info["tags"] = [str(tag) for tag in kwargs["tags"]]
            if (serialized or {}).get("description"):
                context_info["description"] = str(serialized["description"])
            context_info["description"] = describe(
                kind, name, metadata, serialized or {}
            )
            self.contexts[run_id] = context_info
            attrs["lg.context"] = json.dumps(context_info)
            if parent_run_id:
                attrs["lg.parent_run_id"] = str(parent_run_id)
            if kind == "model":
                metadata = kwargs.get("metadata") or {}
                attrs["gen_ai.provider.name"] = (
                    metadata.get("ls_provider") or self.provider_name
                )
                params = kwargs.get("invocation_params") or {}
                effort = (
                    params.get("reasoning_effort")
                    or (params.get("reasoning") or {}).get("effort")
                    or (params.get("output_config") or {}).get("effort")
                    or metadata.get("report_effort")
                )
                if effort:
                    attrs["lg.effort"] = str(effort)
                attrs["gen_ai.request.model"] = (
                    metadata.get("ls_model_name") or self.model_name
                )
            self.spans[run_id] = self.tracer.start_span(
                name, context=context, attributes=attrs
            )
            self.active.add(run_id)

    def _end(self, run_id, error=None, usage=None, model=None, interrupted=False):
        with self.lock:
            if run_id not in self.active:
                return
            span = self.spans[run_id]
            if usage is not None:
                span.set_attribute("lg.usage", json.dumps(usage))
                span.set_attribute("gen_ai.usage.input_tokens", usage["input_tokens"])
                span.set_attribute("gen_ai.usage.output_tokens", usage["output_tokens"])
            if model:
                span.set_attribute("gen_ai.response.model", model)
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
        self._start("workflow", serialized, run_id, parent_run_id, **kwargs)

    def on_chat_model_start(
        self, serialized, messages, *, run_id, parent_run_id=None, **kwargs
    ):
        self._start("model", serialized, run_id, parent_run_id, **kwargs)
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
        with self.lock:
            self.spans[run_id].set_attribute(
                f"lg.{field}", json.dumps(value, default=str)
            )

    def _annotate(self, run_id, values):
        with self.lock:
            self.contexts[run_id].update(values)
            self.spans[run_id].set_attribute(
                "lg.context", json.dumps(self.contexts[run_id])
            )

    def on_llm_start(
        self, serialized, prompts, *, run_id, parent_run_id=None, **kwargs
    ):
        self._start("model", serialized, run_id, parent_run_id, **kwargs)

    def on_tool_start(
        self, serialized, input_str, *, run_id, parent_run_id=None, **kwargs
    ):
        self._start("tool", serialized, run_id, parent_run_id, **kwargs)
        if self.capture_content:
            self._payload(
                run_id, "request", [{"role": "tool_call", "content": input_str}]
            )

    def on_retriever_start(
        self, serialized, query, *, run_id, parent_run_id=None, **kwargs
    ):
        self._start("retriever", serialized, run_id, parent_run_id, **kwargs)

    def on_llm_end(self, response, *, run_id, **kwargs):
        # A chat callback represents one request. Never sum parent graph state messages.
        messages = [
            getattr(g, "message", None) for row in response.generations for g in row
        ]
        if self.capture_content:
            self._payload(run_id, "response", [self._message(m) for m in messages if m])
        usage = next(
            (m.usage_metadata for m in messages if getattr(m, "usage_metadata", None)),
            None,
        )
        model = next(
            (
                m.response_metadata.get("model_name")
                or m.response_metadata.get("model")
                for m in messages
                if m and m.response_metadata
            ),
            None,
        )
        info = {}
        for message in messages:
            if message:
                info.update(message.response_metadata.get("demo_meter", {}))
                if self.capture_content and message.response_metadata.get(
                    "thinking_text"
                ):
                    info["thinking_text"] = str(
                        message.response_metadata["thinking_text"]
                    )
                if message.response_metadata.get("usage_basis"):
                    info["usage_basis"] = message.response_metadata["usage_basis"]
        calls = [
            call["name"] for m in messages if m for call in getattr(m, "tool_calls", [])
        ]
        if calls:
            info["requested_tools"] = calls
        for m in messages:
            if m:
                reason = m.response_metadata.get(
                    "finish_reason"
                ) or m.response_metadata.get("stop_reason")
                if reason:
                    info["finish_reason"] = str(reason)
        self._annotate(run_id, info)
        self._end(run_id, usage=usage, model=model)

    def on_chain_end(self, outputs, *, run_id, **kwargs):
        self._end(
            run_id,
            interrupted=isinstance(outputs, dict)
            and bool(outputs.get("__interrupt__")),
        )

    def on_tool_end(self, output, *, run_id, **kwargs):
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
        self._end(
            run_id,
            error=ToolException()
            if getattr(output, "status", None) == "error"
            else None,
        )

    def on_retriever_end(self, documents, *, run_id, **kwargs):
        self._end(run_id)

    def on_chain_error(self, error, *, run_id, **kwargs):
        self._end(run_id, error=error)

    def on_llm_error(self, error, *, run_id, **kwargs):
        self._end(run_id, error=error)

    def on_tool_error(self, error, *, run_id, **kwargs):
        self._end(run_id, error=error)

    def on_retriever_error(self, error, *, run_id, **kwargs):
        self._end(run_id, error=error)

    def close(self):
        with self.lock:
            for run_id in list(self.active):
                self.spans[run_id].set_attribute("lg.incomplete", True)
                self.spans[run_id].end()
            self.active.clear()
            self.provider.shutdown()
