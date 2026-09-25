"""Construct the configured provider model without invoking it.

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

_logger = logging.getLogger(__name__)
# Identity resolution happens for both reports and adapters, and can happen in
# concurrent browser sessions. Warn once per provider/model for this process.
_warned_models: set[tuple[str, str]] = set()
_warning_lock = Lock()


def configured_identity(
    model_name: str | None = None, *, settings=None
) -> tuple[str, str]:
    """Resolve the allowed model so adapters and accounting use the same identity.

    LG_AVAILABLE_MODELS is an optional comma-separated allowlist for the selected
    provider (deployment names for Azure). Blank means unrestricted. This local
    declaration does not probe provider access or recover from network failures.
    Only an explicitly configured, allowed LG_MODEL can replace a rejected model.
    """
    values = {**(settings or {}), **os.environ}
    provider = values.get("LG_PROVIDER", "openai").lower()
    if provider not in {"openai", "anthropic"}:
        raise ValueError("LG_PROVIDER must be openai or anthropic")
    default = "gpt-5.6-luna" if provider == "openai" else "claude-sonnet-5"
    fallback = (values.get("LG_MODEL") or "").strip()
    requested = model_name or fallback or default
    available = {
        name.strip()
        for name in (values.get("LG_AVAILABLE_MODELS") or "").split(",")
        if name.strip()
    }
    if not available or requested in available:
        return provider, requested

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


def configured_model(model_name: str | None = None, *, settings=None):
    """Give a live sample its configured model and matching accounting identity.

    Return ``(model_adapter, provider, model_id)`` for graph construction and
    reporting; obtaining this tuple does not generate an answer.

    An explicit ``model_name`` overrides LG_MODEL when allowed by
    LG_AVAILABLE_MODELS; otherwise the allowed LG_MODEL is the fallback.
    Read ``LG_PROVIDER`` (``openai``/``anthropic``), ``LG_MODEL``, the matching API key, positive
    LG_MAX_TOKENS, optional LG_EFFORT, and optional OPENAI_BASE_URL from the sample settings overlaid by the process environment.
    No sample writes its .env into global process state. Model access and supported effort values remain the
    provider's responsibility. Missing keys or invalid local settings raise
    ValueError; adapter validation errors also propagate. No request is sent here.

    Each call creates a separate adapter using the same configured model ID.
    The adapter is lazy with respect to network generation: credentials are
    checked locally, but provider-side model/effort validation may still occur
    only when the adapter is first used.
    Different expert roles therefore do not imply different foundation models:
    their system instructions and available tools supply the specialization.
    The returned adapter can be injected into graph construction; the provider
    and model ID strings identify the price entry used by the local reporter.
    """
    values = {**(settings or {}), **os.environ}
    provider, model_id = configured_identity(model_name, settings=values)
    # The validated provider determines which credential is required; an OpenAI
    # key cannot authenticate an Anthropic request, or vice versa.
    key_name = "OPENAI_API_KEY" if provider == "openai" else "ANTHROPIC_API_KEY"
    # Missing, empty, and whitespace-only keys are all unusable. Fail locally
    # before graph execution rather than discovering this after a paid run starts.
    if not values.get(key_name, "").strip():
        raise ValueError(f"Set {key_name} in the environment or .env")
    # Tool arguments can contain complete replacement documents. Give every
    # sample a 32K output allowance without requiring a copied environment file.
    # Explicit settings still let callers match their model or workload limits.
    max_tokens = int(values.get("LG_MAX_TOKENS", "32768"))
    # A nonpositive output budget cannot produce an answer. int() above already
    # rejects nonnumeric settings; this check rejects numeric but unusable values.
    if max_tokens <= 0:
        raise ValueError("LG_MAX_TOKENS must be positive")
    # A nonempty effort value is passed through for SDK/provider validation.
    # Missing or empty means omit the argument and preserve the provider default,
    # rather than assuming all model generations share one effort vocabulary.
    effort_args = (
        {"reasoning_effort": values["LG_EFFORT"]} if values.get("LG_EFFORT") else {}
    )
    # Provider choice selects both the SDK and request shape: OpenAI uses its
    # standard adapter; Anthropic also receives the explicit five-minute cache
    # policy. The earlier allowlist guarantees the else branch means Anthropic.
    # Disable SDK retries so a failed teaching run is visible instead of hiding
    # additional attempts behind one apparent invocation in the report.
    if provider == "openai":
        from langchain_openai import ChatOpenAI

        model_adapter = ChatOpenAI(
            model=model_id,
            # Luna's reasoning + function tools require Responses rather than
            # Chat Completions. Keep the configured effort and real tool calls
            # instead of silently disabling reasoning to make the call pass.
            use_responses_api=True,
            api_key=values[key_name],
            # Sample .env values stay out of global process state, so pass the
            # endpoint explicitly as we do the key. This also supports Azure's
            # OpenAI v1 endpoint. None retains the SDK's normal default/fallback.
            base_url=values.get("OPENAI_BASE_URL") or None,
            max_tokens=max_tokens,
            timeout=60,
            max_retries=0,
            **effort_args,
        )
    else:
        from langchain_anthropic import ChatAnthropic

        from agent_runtime.harness.cache_policy import CACHE_TTL

        model_adapter = ChatAnthropic(
            model_kwargs={"cache_control": {"type": "ephemeral", "ttl": CACHE_TTL}},
            model_name=model_id,
            api_key=values[key_name],
            max_tokens=max_tokens,
            timeout=60,
            max_retries=0,
            **effort_args,
        )
    return model_adapter, provider, model_id
