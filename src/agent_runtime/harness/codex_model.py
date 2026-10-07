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


def parse_result(stdout: str, returncode: int, model: str) -> ChatResult:
    """Require successful terminal evidence and preserve optional usage subsets.

    CLI warnings can be item-level errors before a turn starts. They are not
    failed turns; a nonzero exit or turn.failed is authoritative. Raw stderr and
    error bodies are not propagated because they may contain credentials or
    provider request payloads. No usage receipt means unknown usage, never zero.
    """
    if returncode:
        raise RuntimeError(f"Codex CLI failed (exit {returncode}); check CLI compatibility, login, and model access")
    events = [json.loads(line) for line in stdout.splitlines() if line.strip()]
    if any(event.get("type") in {"turn.failed", "error"} for event in events):
        raise RuntimeError("Codex failed the requested turn")
    completed = [event for event in events if event.get("type") == "turn.completed"]
    if len(completed) != 1:
        raise RuntimeError("Codex did not return exactly one completed turn")
    items = [event["item"] for event in events if event.get("type") == "item.completed"]
    # Graph tools are JSON decisions executed by LangGraph. Native CLI tool
    # work would bypass that authority and invalidates the gateway response.
    if any(item.get("type") not in {"agent_message", "reasoning", "error"} for item in items):
        raise RuntimeError("Codex performed non-text work in request/response mode")
    answers = [item["text"] for item in items if item.get("type") == "agent_message"]
    if not answers:
        raise RuntimeError("Codex completed without an assistant response")
    receipt = completed[0].get("usage") or {}
    # Only wire field names differ. Inclusive accounting and unknown-usage
    # handling are shared with Copilot; retain the original receipt in metadata.
    names = {"cached_input_tokens": "cache_read_input_tokens",
             "cache_write_input_tokens": "cache_creation_input_tokens"}
    return text_result("\n\n".join(answers), {
        "provider": "codex", "model_name": model, "usage": receipt,
        "usage_basis": "Codex CLI turn receipt; model identity is the explicit CLI selection",
    }, usage={names.get(key, key): value for key, value in receipt.items()})


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
        # The minimal profile sends the complete tool contract once in the
        # prompt, using the same validated JSON protocol as Copilot and open
        # schemas. Live schema-only experiments produced duplicate decisions or
        # repeated completed tools. Keep the original strict-format option for
        # callers that have not selected the minimal catalog profile.
        tools, _, _ = self.tool_options(kwargs)
        response_schema = None if self.model_catalog_file else gateway_response_schema(tools)
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
                raise TimeoutError("Codex request timed out") from exc
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
            try:
                stdout, _ = await asyncio.wait_for(process.communicate(prompt), self.request_timeout)
                result = parse_result(stdout.decode("utf-8"), process.returncode, self.model_name)
                if telemetry:
                    result.generations[0].message.response_metadata["codex_telemetry"] = telemetry.evidence()
                return self.gateway_result(result, kwargs)
            finally:
                stop_process(process)
                await process.communicate()
