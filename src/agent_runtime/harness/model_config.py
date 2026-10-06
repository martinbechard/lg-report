"""Construct the configured provider model without invoking it.

Provider construction and lifecycle are delegated through ModelProvider.
Both local-report and Langfuse samples use this factory so changing the tracing
backend does not also change provider settings. Simulation selection belongs to
the caller: invalid live configuration must not become a successful offline run.
Architecture and ownership: docs/chat-composition.md.

AI attribution: Generated with AI assistance.

Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

import logging
import os
from threading import Lock

from .model_providers import get_provider

_logger = logging.getLogger(__name__)
# Identity resolution happens for both reports and adapters, and can happen in
# concurrent browser sessions. Warn once per provider/model for this process.
_warned_models: set[tuple[str, str]] = set()
_warning_lock = Lock()


def configured_identity(
    model_name: str | None = None, *, symbolic_model_name: str | None = None, settings=None
) -> tuple[str, str]:
    """Resolve the allowed model so adapters and accounting use the same identity.

    LG_AVAILABLE_MODELS is an optional comma-separated allowlist for the selected
    provider (deployment names for Azure). Blank means unrestricted. This local
    declaration does not probe provider access or recover from network failures.
    Only an explicitly configured, allowed LG_MODEL can replace a rejected model
    for API providers. Copilot and Codex always require the exact requested model.
    """
    values = {**(settings or {}), **os.environ}
    provider = values.get("LG_PROVIDER", "openai").lower()
    policy = get_provider(provider).policy
    default = policy.default_model
    fallback = (values.get("LG_MODEL") or "").strip()
    # Resolve the symbolic role once at construction, never on each invocation.
    # File settings support the same names as the shell; shell values win above.
    if model_name is not None and symbolic_model_name is not None:
        raise ValueError("Specify either model_name or symbolic_model_name, not both")
    if symbolic_model_name is not None:
        variable = f"LG_MODEL_{symbolic_model_name.upper()}"
        model_name = (values.get(variable) or "").strip()
        if not model_name:
            raise ValueError(
                f"Set {variable} in the environment or .env.local to resolve "
                f"symbolic model {symbolic_model_name!r}"
            )
    requested = model_name or fallback or default
    if policy.exact_model and (not requested or requested == "auto"):
        raise ValueError(f"{provider.title()} requires an explicit LG_MODEL code; auto is not supported")
    available = {
        name.strip()
        for name in (values.get("LG_AVAILABLE_MODELS") or "").split(",")
        if name.strip()
    }
    if not available or requested in available:
        return provider, requested

    if policy.exact_model:
        raise ValueError(f"{provider.title()} model {requested!r} is not in LG_AVAILABLE_MODELS")

    # Never bypass the restriction with an unavailable default, and never pick
    # an arbitrary allowed model: LG_MODEL is the user's fallback decision.
    if not fallback:
        outcome = "No fallback is configured; set LG_MODEL to an allowed model."
    elif fallback not in available:
        outcome = f"Fallback LG_MODEL={fallback!r} is also unavailable."
    else:
        outcome = f"Using fallback LG_MODEL={fallback!r}."
    message = (
        f"Requested model {requested!r} is unavailable for {provider} under "
        f"LG_AVAILABLE_MODELS. {outcome}"
    )
    with _warning_lock:
        if (provider, requested) not in _warned_models:
            _logger.warning(message)
            _warned_models.add((provider, requested))
    if not fallback or fallback not in available:
        raise ValueError(message)
    return provider, fallback


def configured_model(model_name: str | None = None, *, symbolic_model_name: str | None = None, settings=None):
    """Give a live sample its configured model and matching accounting identity.

    Return ``(model_adapter, provider, model_id)`` for graph construction and
    reporting; obtaining this tuple does not generate an answer.

    An explicit ``model_name`` overrides LG_MODEL when allowed by
    LG_AVAILABLE_MODELS; otherwise the allowed LG_MODEL is the fallback.
    Copilot uses the local runtime login and the shared graph-tool gateway protocol. It requires
    an explicit model ID and does not support an output-token override.
    Codex uses an ephemeral local CLI process per request and the existing Codex
    login; LG_CODEX_CLI optionally selects its executable. Neither local adapter
    supports LG_MAX_TOKENS. Both require an explicit model with no fallback.
    Read ``LG_PROVIDER`` (``openai``/``anthropic``/``copilot``/``codex``), ``LG_MODEL``, the matching API key, positive
    LG_MAX_TOKENS, optional LG_EFFORT, and optional OPENAI_BASE_URL from the sample settings overlaid by the process environment.
    No sample writes its .env into global process state. Model access and supported effort values remain the
    provider's responsibility. Missing keys or invalid local settings raise
    ValueError; adapter validation errors also propagate. No request is sent here.

    A symbolic_model_name resolves LG_MODEL_<UPPERCASE_NAME> once during
    construction; a missing or blank mapping names the required variable in an
    error. Supply either a literal model_name or a symbolic_model_name.
    Each call creates a separate adapter using the resolved model ID.
    The adapter is lazy with respect to network generation: credentials are
    checked locally, but provider-side model/effort validation may still occur
    only when the adapter is first used.
    Different expert roles therefore do not imply different foundation models:
    their system instructions and available tools supply the specialization.
    The returned adapter can be injected into graph construction; the provider
    and model ID strings identify the price entry used by the local reporter.
    """
    values = {**(settings or {}), **os.environ}
    provider, model_id = configured_identity(
        model_name, symbolic_model_name=symbolic_model_name, settings=values
    )
    # The protocol owns transport construction. Model selection and all callers
    # remain independent of SDK classes and provider-specific option names.
    return get_provider(provider).create_model(model_id, values), provider, model_id
