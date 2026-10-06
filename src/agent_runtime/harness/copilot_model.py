"""Use Copilot's local SDK server as a tool-free request/response model.

One background event loop owns SDK connections across synchronous CLI and async
web callers. Each invocation gets a fresh session: LangGraph remains the owner
of conversation history. The server is reused while this process runs; shutdown
stops only SDK-owned children. Local connection configuration is never a report.
TextOnlyChatModel owns common input/tracing policy; text_result normalizes usage.
AI attribution: Generated with AI assistance by Avery Northstar.
Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

import asyncio
import json
import os
import secrets
import socket
import threading
from pathlib import Path

from langchain_core.messages import BaseMessage

from .text_model import TextOnlyChatModel, text_result


class CopilotServer:
    """Own one loop and connection so SDK subprocess pipes outlive a request.

    The file records the chosen port and local connection secret atomically.
    An occupied port is reusable only after an SDK handshake, never just because
    something is listening. Startup is serialized inside this process only.
    """

    def __init__(self, config_path: str):
        self.path = Path(config_path).expanduser().resolve()
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever, daemon=True)
        self.thread.start()
        self.client = None
        self.lock = None

    def submit(self, coroutine):
        """Schedule SDK work on its owning loop for either kind of caller."""
        return asyncio.run_coroutine_threadsafe(coroutine, self.loop)

    async def connect(self):
        """Reuse a healthy server or start on the first free port from 7001."""
        from copilot import CopilotClient, RuntimeConnection

        if self.lock is None:
            self.lock = asyncio.Lock()
        async with self.lock:
            if self.client is not None:
                try:
                    await asyncio.wait_for(self.client.ping(), 3)
                    return self.client
                except Exception:  # noqa: BLE001 - SDK transport errors require reconnect
                    # A dead server must not strand later requests on old pipes.
                    await self.client.stop()
                    self.client = None
            saved = json.loads(self.path.read_text()) if self.path.exists() else {}
            port = saved.get("port")
            if port is not None and (type(port) is not int or not 7001 <= port <= 65535):
                raise ValueError(f"Invalid Copilot port in {self.path}")
            # Try the saved endpoint first. Otherwise inspect occupied ports as
            # we scan, allowing an already-running unauthenticated local server.
            candidates = ([port] if port else []) + list(range(7001, 65536))
            seen = set()
            for index, candidate in enumerate(candidates):
                if candidate in seen:
                    continue
                seen.add(candidate)
                with socket.socket() as probe:
                    try:
                        probe.bind(("127.0.0.1", candidate))
                        free = True
                    except OSError:
                        free = False
                if free and index == 0 and candidate == port and candidate != 7001:
                    # A stale saved high port cannot skip newly freed low ports.
                    seen.remove(candidate)
                    continue
                token = saved.get("connection_token") if candidate == port else None
                if free:
                    token = secrets.token_urlsafe(32)
                    connection = RuntimeConnection.for_tcp(
                        port=candidate, connection_token=token
                    )
                else:
                    connection = RuntimeConnection.for_uri(
                        f"127.0.0.1:{candidate}", connection_token=token
                    )
                client = CopilotClient(
                    connection=connection, mode="copilot-cli", use_logged_in_user=True,
                    # Reuse the authenticated Copilot login. Session options below
                    # independently constrain this adapter to request/response.
                    base_directory=str(Path.home() / ".copilot"),
                )
                try:
                    await asyncio.wait_for(client.start(), 120 if free else 3)
                    await asyncio.wait_for(client.ping(), 3)
                except BaseException as exc:
                    await client.stop()
                    if not isinstance(exc, Exception):
                        raise
                    if free:
                        # Retry only a port allocation race, not broken installs
                        # or authentication: those need their real error exposed.
                        with socket.socket() as probe:
                            try:
                                probe.bind(("127.0.0.1", candidate))
                            except OSError:
                                continue
                        raise
                    continue
                try:
                    await asyncio.to_thread(self.save_configuration, candidate, token)
                except BaseException:
                    await client.stop()
                    raise
                self.client = client
                return client
            raise RuntimeError("No free Copilot server port at or above 7001")

    def save_configuration(self, port, token):
        """Publish the selected endpoint only after its RPC handshake succeeds."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        with open(temporary, "w", opener=lambda p, flags: os.open(p, flags, 0o600)) as output:
            json.dump({"port": port, "connection_token": token}, output)
            output.write("\n")
        temporary.replace(self.path)

    async def request(self, model, system, prompt, effort, timeout):
        """Send one turn with no tools or hidden server-side conversation history."""
        from copilot.generated.rpc import PermissionDecisionReject

        client = await self.connect()
        if not (await client.get_auth_status()).isAuthenticated:
            raise ValueError("Copilot is not signed in; run copilot login first")
        models = await client.list_models()
        if model not in {item.id for item in models}:
            raise ValueError(f"Copilot model {model!r} is unavailable; list account models first")
        session = await client.create_session(
            model=model, available_tools=[], tools=[], streaming=False,
            system_message={"mode": "replace", "content": system},
            reasoning_effort=effort,
            infinite_sessions={"enabled": False},
            # Authentication and tool availability are separate decisions. Keep
            # local discovery and side effects off in normal Copilot SDK mode.
            enable_config_discovery=False, skip_custom_instructions=True,
            enable_skills=False, enable_file_hooks=False,
            enable_host_git_operations=False, enable_session_store=False,
            plugin_directories=[], skill_directories=[], mcp_servers={},
            on_permission_request=lambda *_: PermissionDecisionReject(
                feedback="lg-report Copilot requests have tools disabled"
            ),
        )
        usage = []
        # Missing usage stays missing. Pricing discovery uses GitHub's published
        # token rates; the SDK's premium-request multiplier is not a USD charge.
        def observe(event):
            """Keep actual model usage events, including any provider retries."""
            if event.type.value == "assistant.usage":
                usage.append(event.data)

        unsubscribe = session.on(observe)
        try:
            response = await session.send_and_wait(prompt, timeout=timeout)
            if response is None:
                raise RuntimeError("Copilot completed without an assistant response")
            metadata = {"model_name": model, "provider": "copilot"}
            if usage:
                actual = {item.model for item in usage}
                if actual != {model}:
                    raise RuntimeError(f"Copilot returned a different model: {sorted(actual)}")
                # Preserve partial receipts without manufacturing missing counts.
                receipt = {}
                for source, target in [("input_tokens", "input_tokens"),
                                       ("output_tokens", "output_tokens"),
                                       ("cache_read_tokens", "cache_read_input_tokens"),
                                       ("cache_write_tokens", "cache_creation_input_tokens")]:
                    counts = [getattr(item, source, None) for item in usage]
                    if all(value is not None for value in counts):
                        receipt[target] = sum(counts)
                metadata["usage"] = receipt
            return text_result(response.data.content, metadata)
        finally:
            unsubscribe()
            # Delete temporary history even on timeout/provider failure. SDK
            # disconnect detaches this client; delete removes persisted state.
            try:
                await session.disconnect()
            finally:
                await client.delete_session(session.session_id)

    def close(self):
        """Release SDK-owned children at exit; external servers are only detached."""
        try:
            if self.client is not None:
                self.submit(self.client.stop()).result(timeout=15)
                self.client = None
        finally:
            self.loop.call_soon_threadsafe(self.loop.stop)
            self.thread.join(timeout=2)
            self.loop.close()


_servers: dict[str, CopilotServer] = {}
_servers_lock = threading.Lock()


def server_for(path: str) -> CopilotServer:
    """Share one runtime per local configuration without starting it eagerly."""
    key = str(Path(path).expanduser().resolve())
    with _servers_lock:
        if key not in _servers:
            _servers[key] = CopilotServer(key)
        return _servers[key]


def close_copilot_servers():
    """Close at application shutdown, before Python stops its thread executors.

    Direct adapter users must also call this when their application exits.
    Browser turns share servers, so this belongs to app shutdown, not turn cleanup.
    """
    with _servers_lock:
        servers = list(_servers.values())
        _servers.clear()
    for server in servers:
        server.close()


class CopilotChatModel(TextOnlyChatModel):
    """Text-only adapter whose tool binding intentionally disables all tools.

    Graphs may bind their usual tools, but none reach Copilot. Workflows that
    require tool calls are unsuitable for this request/response provider.
    """

    provider = "copilot"
    config_path: str = ".cache/lg-report/copilot.json"

    def _request(self, messages: list[BaseMessage], stop, kwargs):
        """Encode explicit role history; reject inputs the text SDK cannot honor."""
        instructions, history = self.prepare_history(messages, stop, kwargs)
        server = server_for(self.config_path)
        return server.submit(server.request(
            self.model_name, instructions, history,
            self.reasoning_effort, self.request_timeout,
        ))

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        """Bridge synchronous LangChain callers to the persistent SDK loop."""
        return self._request(messages, stop, kwargs).result()

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        """Await without blocking the web event loop; cancellation propagates."""
        return await asyncio.wrap_future(self._request(messages, stop, kwargs))


def main():
    """Print account-specific model names/codes without generating an answer."""
    server = server_for(os.environ.get("LG_COPILOT_CONFIG", ".cache/lg-report/copilot.json"))

    async def listing():
        """Discover through the same auto-start boundary used by model requests."""
        client = await server.connect()
        if not (await client.get_auth_status()).isAuthenticated:
            raise ValueError("Copilot is not signed in; run copilot login first")
        models = await client.list_models()
        if not models:
            raise ValueError("Copilot returned no available models; check login and account model access")
        return [(model.id, model.name) for model in models]

    try:
        for code, name in server.submit(listing()).result():
            print(f"{code}\t{name}")
    finally:
        close_copilot_servers()


if __name__ == "__main__":
    main()
