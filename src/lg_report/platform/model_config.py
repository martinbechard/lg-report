"""Construct the configured provider model without invoking it.

Both local-report and Langfuse samples use this factory so changing the tracing
backend does not also change provider settings. Simulation selection belongs to
the caller: invalid live configuration must not become a successful offline run.
Architecture and ownership: docs/chat-composition.md.

AI attribution: Generated with AI assistance.

Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

import os


def configured_model():
    """Build a LangChain adapter and return it with its provider and model ID.

    Read LG_PROVIDER (openai/anthropic), LG_MODEL, the matching API key, positive
    LG_MAX_TOKENS, and optional LG_EFFORT from the environment already loaded by
    the sample entry point. Model access and supported effort values remain the
    provider's responsibility. Missing keys or invalid local settings raise
    ValueError; adapter validation errors also propagate. No request is sent here.

    Each call creates a separate adapter using the same configured model ID.
    Different expert roles therefore do not imply different foundation models:
    their system instructions and available tools supply the specialization.
    The returned adapter can be injected into graph construction; the provider
    and model ID strings identify the price entry used by the local reporter.
    """
    provider = os.getenv("LG_PROVIDER", "openai").lower()
    # Reject unsupported providers before selecting an SDK or key name. Otherwise
    # any misspelling would silently enter the Anthropic branch below.
    if provider not in {"openai", "anthropic"}:
        raise ValueError("LG_PROVIDER must be openai or anthropic")
    # The validated provider determines which credential is required; an OpenAI
    # key cannot authenticate an Anthropic request, or vice versa.
    key_name = "OPENAI_API_KEY" if provider == "openai" else "ANTHROPIC_API_KEY"
    # Missing, empty, and whitespace-only keys are all unusable. Fail locally
    # before graph execution rather than discovering this after a paid run starts.
    if not os.getenv(key_name, "").strip():
        raise ValueError(f"Set {key_name} in the environment or .env")
    # Use the chosen provider's teaching default only when LG_MODEL is absent.
    # An explicit model ID remains authoritative; this is not a retry fallback.
    default_model_id = "gpt-5.6-luna" if provider == "openai" else "claude-sonnet-5"
    model_id = os.getenv("LG_MODEL", default_model_id)
    max_tokens = int(os.getenv("LG_MAX_TOKENS", "1024"))
    # A nonpositive output budget cannot produce an answer. int() above already
    # rejects nonnumeric settings; this check rejects numeric but unusable values.
    if max_tokens <= 0:
        raise ValueError("LG_MAX_TOKENS must be positive")
    # A nonempty effort value is passed through for SDK/provider validation.
    # Missing or empty means omit the argument and preserve the provider default,
    # rather than assuming all model generations share one effort vocabulary.
    effort_args = (
        {"reasoning_effort": os.environ["LG_EFFORT"]} if os.getenv("LG_EFFORT") else {}
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
            max_tokens=max_tokens,
            timeout=60,
            max_retries=0,
            **effort_args,
        )
    else:
        from langchain_anthropic import ChatAnthropic

        from lg_report.platform.cache_policy import CACHE_TTL

        model_adapter = ChatAnthropic(
            model_kwargs={"cache_control": {"type": "ephemeral", "ttl": CACHE_TTL}},
            model_name=model_id,
            max_tokens=max_tokens,
            timeout=60,
            max_retries=0,
            **effort_args,
        )
    return model_adapter, provider, model_id
