"""Adapt the signed-in Codex CLI to LangChain requests and graph tool decisions.

Each call uses an ephemeral CLI process and temporary working directory, so
LangGraph owns history and no Codex session must be resumed or shared. Commands
use argument vectors, prompts use stdin, and timeouts/cancellation reap the
owned process group. GatewayChatModel shares history/tracing policy with Copilot.
Saved usage is Codex's receipt; reporting applies model rates to estimate cost.
AI attribution: Generated with AI assistance by Ellis Northstar.
Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

import asyncio
import json
import os
import shutil
import signal
import subprocess
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

from langchain_core.outputs import ChatResult

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

    Graph tool calls use the shared JSON gateway protocol; native CLI tools stay
    disabled. Streaming, multimodal input, and output-token limits are unsupported.
    Codex native harness overhead is included in its usage and elapsed time; direct API runs are not identical
    workloads. There is no automatic retry or fallback to another model.
    """

    provider = "codex"
    executable: str = "codex"

    @contextmanager
    def request(self, messages, stop, kwargs):
        """Prepare an isolated, shell-free invocation without changing user config.

        Replace coding instructions with the supplied system messages and encode
        user/assistant history as JSON, as in the Copilot adapter. CLI permission
        rules remain active; normal user config, hooks, and project discovery do
        not contribute instructions or external tools to this model request.
        """
        instructions_text, history = self.prepare_history(messages, stop, kwargs)
        executable = shutil.which(self.executable)
        if executable is None:
            raise ValueError("Codex CLI not found; install Codex and run codex login first")
        with TemporaryDirectory(prefix="lg-report-codex-") as directory:
            instructions = Path(directory) / "instructions.txt"
            instructions.write_text(
                instructions_text + " Do not use native CLI tools or inspect local files. Application tool requests must be returned as JSON decisions.",
                encoding="utf-8",
            )
            command = [executable, "exec", "--ignore-user-config", "--ephemeral", "--json",
                       "--skip-git-repo-check", "--sandbox", "read-only", "--cd", directory,
                       "--model", self.model_name, "-c", "project_doc_max_bytes=0",
                       "-c", 'web_search="disabled"', "-c", "approval_policy=\"never\"",
                       "-c", f"model_instructions_file={json.dumps(str(instructions))}"]
            response_schema = gateway_response_schema(kwargs.get("tools", []))
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
                            "hooks", "view_image"):
                command.extend(["-c", f"features.{feature}=false"])
            command.extend(["-c", "features.skip_host_skill_discovery=true"])
            if self.reasoning_effort:
                command.extend(["-c", f"model_reasoning_effort={json.dumps(self.reasoning_effort)}"])
            command.append("-")
            yield command, history.encode("utf-8")

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        """Block only a synchronous caller and reap the CLI on every exit path."""
        with self.request(messages, stop, kwargs) as (command, prompt):
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE, start_new_session=os.name == "posix")
            try:
                stdout, _ = process.communicate(prompt, timeout=self.request_timeout)
                return self.gateway_result(
                    parse_result(stdout.decode("utf-8"), process.returncode, self.model_name), kwargs
                )
            except subprocess.TimeoutExpired as exc:
                raise TimeoutError("Codex request timed out") from exc
            finally:
                stop_process(process)
                process.communicate()

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        """Let async callers cancel without leaving a model process running."""
        with self.request(messages, stop, kwargs) as (command, prompt):
            process = await asyncio.create_subprocess_exec(
                *command, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE, start_new_session=os.name == "posix",
            )
            try:
                stdout, _ = await asyncio.wait_for(process.communicate(prompt), self.request_timeout)
                return self.gateway_result(
                    parse_result(stdout.decode("utf-8"), process.returncode, self.model_name), kwargs
                )
            finally:
                stop_process(process)
                await process.communicate()
