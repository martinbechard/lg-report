"""Resolve model requests without exposing conversation fixtures to agents.

An application selects a mode for graph construction. The workflow and agents
call build_model with an optional model name and their caller name. Simulation
passes the complete scenario to the simulator, which filters by agent name;
client prompts and documented tool results never become model responses.
Each call creates an independent model. Outside a scope, models are real.
AI attribution: Generated with AI assistance.
Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage

from .model_config import configured_identity, configured_model
from .simulated_model import SimulatedModel

# None means live. A context-local value avoids process-wide mode changes when
# independent browser sessions construct graphs concurrently. Models capture
# their responses at construction and do not consult this scope during a run.
_conversation: ContextVar[Sequence[Mapping[str, Any]] | None] = ContextVar(
    "model_factory_conversation", default=None
)


# Some teaching fixtures consume actual tool observations rather than fixed text.
# Their caller resolver is still selected only inside this construction scope.
_resolver: ContextVar[Callable[[str], BaseChatModel] | None] = ContextVar(
    "scripted_model_resolver", default=None
)


_settings: ContextVar[Mapping[str, str] | None] = ContextVar(
    "model_settings", default=None
)


@contextmanager
def model_factory_scope(
    *,
    live: bool,
    conversation: Sequence[Mapping[str, Any]],
    resolver: Callable[[str], BaseChatModel] | None = None,
    settings: Mapping[str, str] | None = None,
) -> Iterator[tuple[str, str]]:
    """Select construction mode and yield the default report accounting identity.

    Reset even after failed construction or nested scopes. Real mode ignores
    conversation responses and lets provider configuration errors propagate.
    The identity describes the configured default used by this sample; explicit
    per-call model names are still recorded by individual model callbacks.
    """
    token = _conversation.set(None if live else conversation)
    resolver_token = _resolver.set(None if live else resolver)
    settings_token = _settings.set(settings)
    try:
        yield (
            configured_identity(**({"settings": settings} if settings else {}))
            if live
            else ("demo", "scripted-chat")
        )
    finally:
        _conversation.reset(token)
        _resolver.reset(resolver_token)
        _settings.reset(settings_token)


def build_model(model_name: str | None = None, *, caller: str) -> BaseChatModel:
    """Create the requested provider model or the caller's simulated counterpart.

    Omit model_name to use LG_MODEL and the provider default. A workflow-created
    model receives the whole scenario. Native agent identity selects responses
    at invocation, including when several agents share that model. Missing agent
    names fail at invocation. Each construction owns fresh cursors and ledgers.
    """
    conversation = _conversation.get()
    if conversation is None:
        return configured_model(
            model_name, **({"settings": _settings.get()} if _settings.get() else {})
        )[0]
    resolver = _resolver.get()
    if resolver is not None:
        return resolver(caller)
    return SimulatedModel(conversation=list(conversation))


def client_prompts(conversation: Sequence[Mapping[str, Any]]) -> list[str]:
    """Extract static client turns in order without executing any conversation.

    Tool observations are documentation, not canned graph outputs. Only client
    entries become input; named-agent entries remain responses.
    """
    return [entry["content"] for entry in conversation if entry["role"] == "client"]


def model_responses(
    conversation: Sequence[Mapping[str, Any]], speaker: str
) -> list[AIMessage]:
    """Extract one speaker's replies in order, preserving authored usage metadata.

    A chronological script may contain many AI/tool steps between user messages.
    Only this speaker's entries become model replies; tool observations and human
    interruption answers remain outside the model queue. Copies isolate mutable
    messages between runs. Metadata includes illustrative reasoning counts and
    thinking labels used by teaching reports, not evidence of provider billing.
    """
    return [
        AIMessage(
            content=entry["content"],
            tool_calls=entry.get("tool_calls", []),
            response_metadata=entry.get("response_metadata", {}),
        ).model_copy(deep=True)
        for entry in conversation
        if entry["role"] == speaker
    ]
