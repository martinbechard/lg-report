"""Construct the configured provider model without invoking it.

Shared by the training applications and the report CLI so provider timeouts,
cache policy, and environment conventions do not drift between examples.
"""

import os


def configured_model():
    provider = os.getenv("LG_PROVIDER", "openai").lower()
    if provider not in {"openai", "anthropic"}:
        raise ValueError("LG_PROVIDER must be openai or anthropic")
    key_name = "OPENAI_API_KEY" if provider == "openai" else "ANTHROPIC_API_KEY"
    if not os.getenv(key_name, "").strip():
        raise ValueError(f"Set {key_name} in the environment or .env")
    default = "gpt-5.6-luna" if provider == "openai" else "claude-sonnet-5"
    model = os.getenv("LG_MODEL", default)
    max_tokens = int(os.getenv("LG_MAX_TOKENS", "1024"))
    if max_tokens <= 0:
        raise ValueError("LG_MAX_TOKENS must be positive")
    effort_args = (
        {"reasoning_effort": os.environ["LG_EFFORT"]} if os.getenv("LG_EFFORT") else {}
    )
    if provider == "openai":
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            model=model,
            max_tokens=max_tokens,
            timeout=60,
            max_retries=0,
            **effort_args,
        )
    else:
        from langchain_anthropic import ChatAnthropic

        from .cache_policy import CACHE_TTL

        llm = ChatAnthropic(
            model_kwargs={"cache_control": {"type": "ephemeral", "ttl": CACHE_TTL}},
            model_name=model,
            max_tokens=max_tokens,
            timeout=60,
            max_retries=0,
            **effort_args,
        )
    return llm, provider, model
