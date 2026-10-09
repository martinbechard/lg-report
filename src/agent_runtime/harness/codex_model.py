"""Adapt the signed-in Codex CLI to LangChain requests and graph tool decisions.

Each call uses an ephemeral CLI process and temporary working directory, so
LangGraph owns history and no Codex session must be resumed or shared. Commands
use argument vectors, prompts use stdin, and timeouts/cancellation reap the
owned process group. GatewayChatModel shares history/tracing policy with Copilot.
Saved usage is Codex's receipt; reporting applies model rates to estimate cost.
Native timing metrics are collected on a per-call loopback OTLP receiver.
AI attribution: Generated with AI assistance by Ellis Northstar.
Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

import asyncio
import json
import os
import shutil
import signal
import subprocess
from contextlib import contextmanager, nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory

from langchain_core.outputs import ChatResult

from .codex_telemetry import CodexTelemetry
from .gateway_model import GatewayChatModel, gateway_response_schema, text_result


class CodexProcessError(RuntimeError):
    """Expose a safe failure category while retaining any completed receipt."""

    def __init__(self, message, result=None):
        """Keep optional metering separate from the failed decision's usability."""
        super().__init__(message)
        self.result = result


class CodexCapacityError(CodexProcessError):
    """The provider explicitly rejected the selected model as at capacity."""


class CodexRequestTimeout(TimeoutError):
    """A bounded CLI call expired, possibly after producing a usage receipt."""

    def __init__(self, result=None):
        """Never imply that killing an unfinished call makes it free."""
        super().__init__("Codex request timed out")
        self.result = result


def completed_result(events, model):
    """Extract one authoritative receipt without deciding whether work succeeded.

    A shutdown failure or rejected native action must not erase paid inference.
    Multiple or absent terminal receipts remain unknown rather than guessing
    which totals are inclusive. Callers separately validate process/protocol
    success before permitting a response to reach graph tools.
    """
    completed = [event for event in events if event.get("type") == "turn.completed"]
    if len(completed) != 1:
        return None
    answers = [event["item"]["text"] for event in events
               if event.get("type") == "item.completed"
               and event.get("item", {}).get("type") == "agent_message"]
    receipt = completed[0].get("usage") or {}
    names = {"cached_input_tokens": "cache_read_input_tokens",
             "cache_write_input_tokens": "cache_creation_input_tokens"}
    # SDK finalResponse semantics select the last agent message. Intermediate
    # commentary remains charged by the full receipt, never executed as a tool.
    return text_result(answers[-1] if answers else "", {
        "provider": "codex", "model_name": model, "usage": receipt,
        "usage_basis": "Codex CLI turn receipt; model identity is the explicit CLI selection",
    }, usage={names.get(key, key): value for key, value in receipt.items()})


def timeout_result(stdout, model):
    """Recover only complete JSONL receipts after terminating a timed-out CLI.

    Killing a process can cut its final line in half. Ignore that incomplete
    line; preceding complete receipts remain evidence. No partial token counts
    or text-based token estimates are substituted for provider totals.
    """
    events = []
    for line in stdout.decode("utf-8", errors="replace").splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if isinstance(event, dict):
            events.append(event)
    return completed_result(events, model)


def parse_result(stdout: str, returncode: int, model: str) -> ChatResult:
    """Require successful terminal evidence, retaining usage even on failure.

    Error bodies and stderr can contain credentials or payloads. Only known
    capacity wording is classified; other errors use a fixed safe description.
    Startup item warnings do not invalidate an otherwise successful turn.
    """
    try:
        events = [json.loads(line) for line in stdout.splitlines() if line.strip()]
    except ValueError:
        if returncode:
            raise CodexProcessError(f"Codex CLI failed (exit {returncode})") from None
        raise
    result = completed_result(events, model)
    failures = [event for event in events if event.get("type") in {"turn.failed", "error"}]
    if any("selected model is at capacity" in str(event.get("message", "")).lower()
           or "selected model is at capacity" in str(event.get("error", {}).get("message", "")).lower()
           for event in failures):
        raise CodexCapacityError("Selected Codex model is at capacity", result)
    if returncode:
        raise CodexProcessError(f"Codex CLI failed (exit {returncode})", result)
    if failures:
        raise CodexProcessError("Codex failed the requested turn", result)
    if result is None:
        raise CodexProcessError("Codex did not return exactly one completed turn")
    items = [event["item"] for event in events if event.get("type") == "item.completed"]
    # Native actions bypass graph ownership and must fail even when metered.
    if any(item.get("type") not in {"agent_message", "reasoning", "error"} for item in items):
        raise CodexProcessError("Codex performed non-text work in request/response mode", result)
    if not any(item.get("type") == "agent_message" for item in items):
        raise CodexProcessError("Codex completed without an agent response", result)
    return result


def stop_process(process):
    """Terminate the owned CLI and its descendants on timeout or cancellation.

    POSIX groups include the Node launcher and its native CLI child. On Windows,
    the configured executable should be the native CLI so killing it is direct.
    The caller still drains pipes and waits to reap the process before cleanup.
    """
    if process.returncode is None:
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
        except ProcessLookupError:
            pass  # It exited between the returncode check and the kill.


class CodexChatModel(GatewayChatModel):
    """Run model decisions through Codex using its existing local authentication.

    Graph tool calls use the shared JSON gateway protocol; native CLI actions
    are prohibited and rejected if observed. The optional minimal profile also
    removes their definitions using a copy of exact model metadata. Streaming,
    multimodal input, and output-token limits are unsupported.
    Codex native harness overhead is included in its usage and elapsed time; direct API runs are not identical
    workloads. There is no automatic retry or fallback to another model.
    """

    provider = "codex"
    executable: str = "codex"
    capture_telemetry: bool = True
    model_catalog_file: str | None = None

    def gateway_catalog(self):
        """Copy selected model metadata with native tool advertisement removed.

        The optional source is a Codex models cache or catalog. Requiring an
        exact entry avoids inventing model capabilities or changing reasoning,
        routing and context settings. Only the temporary copy changes; account
        metadata and other models from a cache are never forwarded to the CLI.
        """
        if self.model_catalog_file is None:
            return None
        catalog = json.loads(Path(self.model_catalog_file).expanduser().read_text(encoding="utf-8"))
        matches = [entry for entry in catalog["models"] if entry.get("slug") == self.model_name]
        if len(matches) != 1:
            raise ValueError("Codex model catalog must contain exactly one entry for the selected model")
        model = {**matches[0], "tool_mode": "direct", "shell_type": "disabled",
                 "apply_patch_tool_type": None, "experimental_supported_tools": []}
        return {"models": [model]}

    @contextmanager
    def request(self, messages, stop, kwargs):
        """Prepare an isolated, shell-free invocation without changing user config.

        Replace coding instructions with the supplied system messages and encode
        user/assistant history as JSON, as in the Copilot adapter. CLI permission
        rules remain active. Config and project discovery are suppressed, but
        global AGENTS.md can still apply. Explicit switches suppress automatic
        skills; the optional catalog profile also removes model-advertised
        native tools.
        """
        # Removing native CLI tools must not remove the gateway's output
        # contract. Prompt-only Luna responses mixed task JSON with commentary
        # and omitted the envelope. Keep the full history/protocol instructions
        # alongside the schema: the schema constrains shape, not tool semantics.
        tools, _, _ = self.tool_options(kwargs)
        response_schema = gateway_response_schema(tools)
        instructions_text, history = self.prepare_history(messages, stop, kwargs)
        catalog = self.gateway_catalog()
        executable = shutil.which(self.executable)
        if executable is None:
            raise ValueError("Codex CLI not found; install Codex and run codex login first")
        with TemporaryDirectory(prefix="lg-report-codex-") as directory:
            instructions = Path(directory) / "instructions.txt"
            instructions.write_text(
                instructions_text + ("" if tools else " Do not use native CLI tools or inspect local files."),
                encoding="utf-8",
            )
            command = [executable, "exec", "--ignore-user-config", "--ephemeral", "--json",
                       "--skip-git-repo-check", "--sandbox", "read-only", "--cd", directory,
                       "--model", self.model_name, "-c", "project_doc_max_bytes=0",
                       "-c", 'web_search="disabled"', "-c", "approval_policy=\"never\"",
                       "-c", f"model_instructions_file={json.dumps(str(instructions))}"]
            if catalog is not None:
                # Model metadata can override ordinary feature switches. This
                # startup-only catalog removes the remaining native tools while
                # preserving every unrelated capability from the supplied entry.
                catalog_path = Path(directory) / "model-catalog.json"
                catalog_path.write_text(json.dumps(catalog), encoding="utf-8")
                command.extend(["-c", f"model_catalog_json={json.dumps(str(catalog_path))}"])
                # The model has no CLI tools to use. Omit their textual sandbox
                # briefing, while keeping the actual read-only sandbox and
                # approval policy enforced by the CLI invocation above.
                command.extend(["-c", "include_permissions_instructions=false"])
            if response_schema is not None:
                # The CLI can constrain its final response shape. The shared
                # validator still checks names and choices; the graph validates
                # arguments before executing tools under its own permissions.
                schema = Path(directory) / "response-schema.json"
                schema.write_text(json.dumps(response_schema), encoding="utf-8")
                command.extend(["--output-schema", str(schema)])
            # These are native CLI feature switches, not a second tool runtime.
            # Keeping host extensions off avoids invoking personal integrations.
            for feature in ("shell_tool", "apps", "plugins", "memories", "multi_agent",
                            "browser_use", "computer_use", "image_generation", "skill_search",
                            "hooks", "view_image", "goals", "multi_agent_v2", "sleep_tool"):
                command.extend(["-c", f"features.{feature}=false"])
            command.extend(["-c", "features.skip_host_skill_discovery=true"])
            # Discovery and prompt inclusion are different CLI controls. Wire
            # inspection showed that skipping discovery alone still included a
            # large skill catalog. These switches remove unrelated gateway
            # context and native question/goal tools, preserving sandbox rules.
            for setting in ("skills.include_instructions=false",
                            "agents.enabled=false",
                            "tools.experimental_request_user_input.enabled=false",
                            "include_environment_context=false",
                            "include_collaboration_mode_instructions=false",
                            "include_apps_instructions=false"):
                command.extend(["-c", setting])
            # Sol's model profile still injected delegation tools and role
            # descriptions with only features.multi_agent=false. The current
            # agents.enabled control removes that payload; V2 must also stay
            # off because it can override the agent setting. The catalog profile
            # handles model-native exec/wait definitions separately.
            if self.reasoning_effort:
                command.extend(["-c", f"model_reasoning_effort={json.dumps(self.reasoning_effort)}"])
            command.append("-")
            yield command, history.encode("utf-8")

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        """Block only a synchronous caller and reap the CLI on every exit path."""
        with self.request(messages, stop, kwargs) as (command, prompt), (
            CodexTelemetry() if self.capture_telemetry else nullcontext()
        ) as telemetry:
            if telemetry:
                command = command[:-1] + telemetry.arguments() + command[-1:]
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE, start_new_session=os.name == "posix")
            try:
                stdout, _ = process.communicate(prompt, timeout=self.request_timeout)
                result = parse_result(stdout.decode("utf-8"), process.returncode, self.model_name)
                if telemetry:
                    result.generations[0].message.response_metadata["codex_telemetry"] = telemetry.evidence()
                return self.gateway_result(result, kwargs)
            except subprocess.TimeoutExpired as exc:
                # communicate can be called again after killing the process. Its
                # returned stdout contains everything, including buffered lines.
                stop_process(process)
                stdout, _ = process.communicate()
                raise CodexRequestTimeout(timeout_result(stdout, self.model_name)) from exc
            finally:
                stop_process(process)
                process.communicate()

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        """Let async callers cancel without leaving a model process running."""
        with self.request(messages, stop, kwargs) as (command, prompt), (
            CodexTelemetry() if self.capture_telemetry else nullcontext()
        ) as telemetry:
            if telemetry:
                command = command[:-1] + telemetry.arguments() + command[-1:]
            process = await asyncio.create_subprocess_exec(
                *command, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE, start_new_session=os.name == "posix",
            )
            # Shield the single pipe reader: cancelling communicate discards
            # bytes it already consumed, so a second call cannot recover them.
            communication = asyncio.create_task(process.communicate(prompt))
            try:
                stdout, _ = await asyncio.wait_for(asyncio.shield(communication), self.request_timeout)
                result = parse_result(stdout.decode("utf-8"), process.returncode, self.model_name)
                if telemetry:
                    result.generations[0].message.response_metadata["codex_telemetry"] = telemetry.evidence()
                return self.gateway_result(result, kwargs)
            except TimeoutError as exc:
                stop_process(process)
                stdout, _ = await communication
                raise CodexRequestTimeout(timeout_result(stdout, self.model_name)) from exc
            finally:
                stop_process(process)
                await communication
