"""Simulate one shared LLM while retaining separate scripted agent conversations.

A provider accepts messages and tools on each call; it does not own agent history.
This offline counterpart chooses its script by the bound tool name, or the
system instruction for a role without tools.
The dispatcher/expert roles have distinct tools, so no sample prompts or role
imports belong here. Accounting reuses the ordinary simulator's implementation.
AI attribution: Generated with AI assistance.

Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

from threading import Lock

from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import Field, PrivateAttr

from .demo_meter import ContextSimulation
from .simulated_model import ScriptedChatModel, meter_response


class SharedSimulatedModel(ScriptedChatModel):
    """One LLM fixture with a response script per distinct bound tool set.

    scripts maps a tool name (or a tool-free role's system instruction) to authored
    AI responses. Unsupported tool sets fail explicitly. State is scoped by that
    key for teaching scenarios, not a general multi-session cache implementation.
    """

    scripts: dict[str, list[AIMessage]]
    responses: list[AIMessage] = Field(default_factory=list)
    _positions: dict[str, int] = PrivateAttr(default_factory=dict)
    _contexts: dict[str, ContextSimulation] = PrivateAttr(default_factory=dict)
    _lock: Lock = PrivateAttr(default_factory=Lock)

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        """Give each agent its own tool binding while sharing one scripted model.

        Each graph retains its own binding. Mutating the shared model here would
        let the last expert overwrite the dispatcher available task tool.
        ``tools`` is converted into the callback-visible OpenAI schema; the
        ``tool_choice`` and other keyword options remain fixture-ignored because
        authored responses, rather than provider selection, control this fake.
        """
        schemas = [convert_to_openai_tool(tool) for tool in tools]
        return self.bind(tool_definitions=schemas)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        """Answer one offline agent turn using that role's authored response script.

        ``messages`` is the current LangChain transcript. Bound tool schemas
        arrive through ``kwargs``; ``stop`` and ``run_manager`` are framework
        parameters unused by this fixture. Return a LangChain ChatResult whose
        generation wraps an AIMessage, possibly proposing a tool call. The graph
        executes any proposed tool after this model method returns.

        A one-tool binding is keyed by its function name; a tool-free binding is
        keyed by its first system instruction. Position, context, and exhaustion
        are protected by one lock so concurrent graph calls cannot reuse a script
        item. Unknown keys and exhausted scripts raise explicitly.
        """
        schemas = kwargs.get("tool_definitions", [])
        if schemas:
            # Tool-based examples have exactly one tool per role; reject ambiguous
            # bindings rather than consume a different agent's scripted answer.
            if len(schemas) != 1:
                raise ValueError("Shared fixture requires one bound tool per agent")
            script_key = schemas[0]["function"]["name"]
        else:
            # Tool-free roles (author/judge) are distinguished by their actual
            # system instruction. Scripts remain in the test case, not the agent.
            script_key = str(messages[0].content)
        with self._lock:
            position = self._positions.get(script_key, 0)
            script = self.scripts[script_key]
            # Exhaustion is a broken scenario, not permission to replay answers.
            if position >= len(script):
                raise ValueError(f"No scripted response left for {script_key}")
            # Copy the AIMessage fixture so per-call usage cannot contaminate
            # the authored answer shared by later test runs.
            response = script[position].model_copy(deep=True)
            context = self._contexts.setdefault(script_key, ContextSimulation())
            response = meter_response(response, context, schemas, messages)
            self._positions[script_key] = position + 1
        return ChatResult(generations=[ChatGeneration(message=response)])
