"""Serve a small Angular chat using standard AG-UI HTTP/SSE streaming.

The application selects sample factories and frames text attachments. The
AG-UI LangGraph adapter, event encoder and browser HttpAgent own streaming.
The shared agent_runtime launcher owns CLI parsing and configuration; this
module defines HTTP handlers and starts the listener with prepared settings.
This is a loopback-only, single-user demonstration app, not a hosted service.
AI attribution: Generated with AI assistance by Northstar.
Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from time import monotonic
from uuid import UUID, uuid4

from ag_ui.core import RunAgentInput, UserMessage
from ag_ui.encoder import EventEncoder
from fastapi import FastAPI, HTTPException
from fastapi import Request as HttpRequest
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from starlette.middleware.trustedhost import TrustedHostMiddleware

from agent_runtime.harness.conversation import Attachment, Request
from agent_runtime.harness.sample_catalog import SampleCatalog
from agent_runtime.harness.web_conversation import WebConversation

logger = logging.getLogger(__name__)

MAX_REQUEST_BYTES = 8 * 1024 * 1024


class Selection(BaseModel):
    """Select a sample and execution mode; model keys stay server-configured."""

    model_config = ConfigDict(extra="forbid", strict=True)
    sample: str
    live: bool | None = None


class FileInput(BaseModel):
    """Read text and a display filename, matching the console's Attachment."""

    model_config = ConfigDict(extra="forbid", strict=True)
    name: str = Field(min_length=1)
    content: str


class Turn(BaseModel):
    """Carry the current user request in standard AG-UI forwarded properties."""

    model_config = ConfigDict(extra="forbid", strict=True)
    prompt: str = ""
    attachments: list[FileInput] = Field(default_factory=list)


def create_app(
    *,
    directory=Path("reports"),
    ui_directory=None,
    prices=None,
    factory=None,
    catalog=None,
    default_sample="simple_chat",
    default_live=False,
    capture_content=False,
    prompt_overrides=None,
    sample_directory=None,
):
    """Build a local app; every chat retains its own factory/model instance.

    HTTP session creation replaces socket handshakes. Inactive sessions expire
    after an hour; New chat explicitly deletes the old one. The busy flag only
    prevents overlapping turns from corrupting a single conversation's cursor.
    sample_directory lets a launched sample honor its exact --out directory;
    other catalog samples use their own folder beneath directory.
    """
    catalog = catalog or SampleCatalog()
    if not catalog.samples:
        raise ValueError("No samples discovered")
    if default_sample not in catalog.samples:
        default_sample = next(iter(catalog.samples))
    factory = factory or catalog.create_run
    sessions = {}

    @asynccontextmanager
    async def lifespan(app):
        # Server shutdown follows in-flight response teardown. Each session
        # then releases its exporter and isolated file workspace exactly once.
        try:
            yield
        finally:
            for session in sessions.values():
                await asyncio.to_thread(session.close)
            sessions.clear()

    app = FastAPI(title="Sample chat", lifespan=lifespan)
    app.add_middleware(
        TrustedHostMiddleware, allowed_hosts=["localhost", "127.0.0.1", "[::1]"]
    )
    app.state.sessions = sessions

    @app.middleware("http")
    async def local_requests(request: HttpRequest, call_next):
        """Keep state-changing requests same-origin without adding user accounts."""
        from fastapi.responses import JSONResponse

        if request.method not in {"GET", "HEAD", "OPTIONS"}:
            origin = request.headers.get("origin")
            allowed = {str(request.base_url).rstrip("/"), "http://localhost:4200"}
            if origin and origin not in allowed:
                return JSONResponse({"detail": "Local requests only."}, status_code=403)
        return await call_next(request)

    def info(sample):
        """Preserve launch-supplied example input without changing agent prompts."""
        result = catalog.info(sample)
        if prompt_overrides and sample in prompt_overrides:
            result["prompts"] = prompt_overrides[sample]
        return result

    @app.get("/api/samples")
    async def samples():
        """Return menu labels, example prompts and the launcher default mode."""
        return {
            "samples": [info(key) for key in catalog.samples],
            "maxRequestBytes": MAX_REQUEST_BYTES,
            "defaultSample": default_sample,
            "live": default_live,
        }

    @app.post("/api/sessions")
    async def start(selection: Selection):
        """Create the same sample as the console without invoking its model."""
        if selection.sample not in catalog.samples:
            raise HTTPException(404, "Unknown sample")
        for key, value in list(sessions.items()):
            if not value.busy and monotonic() - value.touched > 3600:
                await asyncio.to_thread(value.close)
                del sessions[key]
        # Mode belongs to the new session. Changing it creates fresh graph
        # state so scripted cursors and real model history cannot be mixed.
        live = default_live if selection.live is None else selection.live
        try:
            graph, provider, model = await asyncio.to_thread(
                factory, selection.sample, live
            )
        except Exception:
            logger.exception("Cannot construct sample %s", selection.sample)
            raise HTTPException(
                503, "Cannot open this sample. Check the server configuration."
            ) from None
        thread_id = str(uuid4())
        sessions[thread_id] = WebConversation(
            graph,
            provider,
            model,
            sample=selection.sample,
            live=live,
            directory=(
                sample_directory
                if selection.sample == default_sample and sample_directory is not None
                else Path(directory) / selection.sample
            ),
            prices=prices,
            capture_content=capture_content,
        )
        return {"threadId": thread_id}

    @app.delete("/api/sessions/{thread_id}", status_code=204)
    async def discard(thread_id: UUID):
        """Drop inactive model/checkpoint state after a user starts a new chat."""
        session = sessions.get(str(thread_id))
        if session and session.busy:
            raise HTTPException(409, "Wait for the current response to stop.")
        if session:
            await asyncio.to_thread(session.close)
        sessions.pop(str(thread_id), None)

    @app.get("/api/sessions/{thread_id}/context")
    async def context_evidence(thread_id: UUID):
        """Export the separate claims audit under the configured capture policy."""
        session = sessions.get(str(thread_id))
        context = getattr(session.workflow, "context_audit", None) if session else None
        if context is None:
            raise HTTPException(404, "This session has no context audit.")
        return context.evidence()

    @app.head("/api/sessions/{thread_id}/output")
    @app.get("/api/sessions/{thread_id}/output")
    async def output(thread_id: UUID):
        """Download only this session's configured output, never a browser path."""
        session = sessions.get(str(thread_id))
        path = getattr(session.workflow, "output_file", None) if session else None
        if path is None or not path.is_file():
            raise HTTPException(404, "No approved output exists for this session.")
        return FileResponse(
            path, filename="edited-summary.txt", media_type="text/plain"
        )

    @app.post("/api/chat/{thread_id}")
    async def chat(thread_id: UUID, request: HttpRequest):
        """Accept RunAgentInput and let the official encoder format its SSE events.

        The checkpoint is authoritative. Only the new human message and shared
        Request formatting enter the graph; browser state/tool definitions and
        resume answers are matched to pending interrupts before execution.
        """
        session = sessions.get(str(thread_id))
        if session is None or not session.usable:
            raise HTTPException(409, "Start a new chat to continue.")
        if session.busy:
            raise HTTPException(409, "A response is already running.")
        body = await request.body()
        if len(body) > MAX_REQUEST_BYTES:
            raise HTTPException(413, "Message and attachments exceed 8 MiB.")
        try:
            data = RunAgentInput.model_validate_json(body)
            UUID(data.run_id)
            turn = Turn.model_validate(data.forwarded_props)
        except (ValidationError, ValueError):
            raise HTTPException(422, "Invalid chat request") from None
        if data.thread_id != str(thread_id):
            raise HTTPException(422, "Thread identity does not match the session.")
        if data.resume:
            # Only responses to the actual pending IDs can continue this thread.
            # Resume is not a new turn and must not replay browser message history.
            ids = [entry.interrupt_id for entry in data.resume]
            if len(ids) != len(set(ids)) or set(ids) != {
                item.id for item in session.pending
            }:
                raise HTTPException(
                    422, "Resume must answer each pending interaction exactly once."
                )
            if any(entry.status != "resolved" for entry in data.resume):
                raise HTTPException(
                    422, "Send the workflow cancellation choice as a resolved response."
                )
            data = data.model_copy(
                update={
                    "messages": [],
                    "state": {},
                    "tools": [],
                    "context": [],
                    "forwarded_props": {},
                }
            )
        else:
            if session.pending:
                raise HTTPException(
                    409,
                    "Answer the pending interaction before sending another request.",
                )
            if not data.messages or data.messages[-1].role != "user":
                raise HTTPException(422, "A new user message is required.")
            if not turn.prompt.strip() and not turn.attachments:
                raise HTTPException(422, "Enter a message or attach a text file.")
            if not session.live:
                prompts = info(session.sample)["prompts"]
                if session.turn >= len(prompts) or turn.prompt != prompts[session.turn]:
                    raise HTTPException(
                        422,
                        "This example uses the supplied prompts. Select Use real LLM for free-form chat.",
                    )
            content = Request(
                turn.prompt,
                tuple(Attachment(f.name, f.content) for f in turn.attachments),
            ).content()
            data = data.model_copy(
                update={
                    "messages": [UserMessage(id=data.messages[-1].id, content=content)],
                    "state": {},
                    "tools": [],
                    "context": [],
                    "forwarded_props": {},
                }
            )
        session.busy = True
        encoder = EventEncoder()

        async def events():
            """Use the library's SSE encoding and the HTTP response's cancellation."""
            from contextlib import aclosing

            async with aclosing(session.run(data)) as stream:
                async for event in stream:
                    yield encoder.encode(event)

        return StreamingResponse(
            events(),
            media_type=encoder.get_content_type(),
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    if ui_directory and Path(ui_directory).is_dir():
        app.mount("/", StaticFiles(directory=ui_directory, html=True), name="chat-ui")
    return app


def start_workflow_api_listener(
    *, catalog, sample_id, settings, port, options=None, prompts=None, env_file=None
):
    """Serve the discovered catalog, constructing a fresh workflow per session."""
    import uvicorn

    import samples

    def factory(selected, live):
        return catalog.create_run(
            selected,
            live,
            options=options if selected == sample_id else None,
            env_file=env_file,
        )

    root = Path(samples.__file__).resolve().parent.parent
    app = create_app(
        catalog=catalog,
        factory=factory,
        directory=settings.output.parent,
        sample_directory=settings.output,
        ui_directory=root / "frontend/dist/chat/browser",
        prices=settings.prices,
        default_sample=sample_id,
        default_live=settings.live,
        capture_content=settings.capture_content,
        prompt_overrides={sample_id: prompts} if prompts is not None else None,
    )
    print(f"Angular chat: http://127.0.0.1:{port}")
    uvicorn.run(app, host="127.0.0.1", port=port)
