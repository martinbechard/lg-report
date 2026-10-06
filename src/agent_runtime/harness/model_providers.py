"""Define the provider boundary used by model configuration and app shutdown.

A provider constructs a LangChain model; BaseChatModel remains the invocation
contract so native API tool calling, streaming, and multimodal input survive.
The registry owns configuration policy, not pricing: all receipts still flow to
reporting's shared cost estimator. Imports and runtime startup remain lazy.
AI attribution: Generated with AI assistance by Ellis Northstar.
Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from langchain_core.language_models import BaseChatModel


@dataclass(frozen=True)
class ProviderPolicy:
    """Declare selection and credential rules once for every application entry.

    Local login providers have no API credential variable and require an exact
    model. Output limits are supported only when the transport can honor them.
    """

    name: str
    default_model: str = ""
    credential_variable: str | None = None
    exact_model: bool = True
    supports_max_tokens: bool = False

    def is_configured(self, settings: Mapping[str, str]) -> bool:
        """Detect live mode locally, without probing login or model access."""
        return self.credential_variable is None or bool(
            (settings.get(self.credential_variable) or "").strip()
        )


class ModelProvider(Protocol):
    """Construction/lifetime boundary; applications never select SDK classes.

    Implementations return an uninvoked model and close only shared resources
    they own. Per-request cleanup remains the transport's responsibility.
    """

    @property
    def policy(self) -> ProviderPolicy:
        """Expose configuration rules without importing or starting an SDK."""
        ...

    def create_model(self, model: str, settings: Mapping[str, str]) -> BaseChatModel:
        """Construct the exact resolved model, propagating configuration errors."""
        ...

    def close(self) -> None:
        """Release owned process-wide resources at application shutdown."""
        ...


@dataclass(frozen=True)
class RegisteredProvider:
    """Apply common option validation before a small SDK-specific constructor.

    Composition keeps four providers from repeating credential, output budget,
    and effort handling. A different implementation may satisfy ModelProvider
    directly when it cannot use this constructor shape.
    """

    policy: ProviderPolicy
    build: Callable[[str, Mapping[str, str], dict[str, Any]], BaseChatModel]
    shutdown: Callable[[], None] | None = None

    def create_model(self, model: str, settings: Mapping[str, str]) -> BaseChatModel:
        """Validate supported options before any transport is constructed."""
        options: dict[str, Any] = {}
        key = self.policy.credential_variable
        if key:
            if not self.policy.is_configured(settings):
                raise ValueError(f"Set {key} in the environment or .env")
            options["api_key"] = settings[key]
        if self.policy.supports_max_tokens:
            # Replacement documents can be large; preserve the common 32K
            # allowance while rejecting a budget that cannot produce an answer.
            limit = int(settings.get("LG_MAX_TOKENS", "32768"))
            if limit <= 0:
                raise ValueError("LG_MAX_TOKENS must be positive")
            options["max_tokens"] = limit
        elif settings.get("LG_MAX_TOKENS"):
            raise ValueError(f"{self.policy.name.title()} does not expose LG_MAX_TOKENS; unset it")
        # Omission preserves each provider's default and effort vocabulary.
        if settings.get("LG_EFFORT"):
            options["reasoning_effort"] = settings["LG_EFFORT"]
        return self.build(model, settings, options)

    def close(self) -> None:
        """Skip providers whose models own no shared background runtime."""
        if self.shutdown is not None:
            self.shutdown()


def _api_options(options: dict[str, Any]) -> dict[str, Any]:
    """Make failed API attempts visible instead of hiding retries in one span."""
    return {**options, "timeout": 60, "max_retries": 0}


def _openai(model, settings, options):
    """Keep native Responses tools/reasoning and explicit custom endpoints."""
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model=model, use_responses_api=True,
                      base_url=settings.get("OPENAI_BASE_URL") or None,
                      **_api_options(options))


def _anthropic(model, settings, options):
    """Keep native Anthropic behavior with the shared ephemeral cache policy."""
    from langchain_anthropic import ChatAnthropic

    from .cache_policy import CACHE_TTL

    return ChatAnthropic(model_name=model,
                         model_kwargs={"cache_control": {"type": "ephemeral", "ttl": CACHE_TTL}},
                         **_api_options(options))


def _codex(model, settings, options):
    """Defer CLI discovery/login checks until the first request."""
    from .codex_model import CodexChatModel

    return CodexChatModel(model_name=model, executable=settings.get("LG_CODEX_CLI") or "codex",
                          **options)


def _copilot(model, settings, options):
    """Check the optional installation without starting a Copilot server."""
    try:
        import copilot  # noqa: F401 - validate the optional install locally
    except ImportError as exc:
        raise ValueError("Install Copilot support with uv sync --extra copilot") from exc
    from .copilot_model import CopilotChatModel

    return CopilotChatModel(model_name=model,
                            config_path=settings.get("LG_COPILOT_CONFIG") or ".cache/lg-report/copilot.json",
                            **options)


def _close_copilot():
    """Keep optional runtime knowledge behind the provider boundary."""
    from .copilot_model import close_copilot_servers

    close_copilot_servers()


# One entry supplies selection, construction, and shared shutdown behavior.
# Workflow code needs no new branch when a provider is added here.
PROVIDERS: dict[str, ModelProvider] = {
    "openai": RegisteredProvider(ProviderPolicy("openai", "gpt-5.6-luna", "OPENAI_API_KEY",
                                               exact_model=False, supports_max_tokens=True), _openai),
    "anthropic": RegisteredProvider(ProviderPolicy("anthropic", "claude-sonnet-5", "ANTHROPIC_API_KEY",
                                                  exact_model=False, supports_max_tokens=True), _anthropic),
    "codex": RegisteredProvider(ProviderPolicy("codex"), _codex),
    "copilot": RegisteredProvider(ProviderPolicy("copilot"), _copilot, _close_copilot),
}


def get_provider(name: str) -> ModelProvider:
    """Reject unknown providers before selection or credential inference."""
    try:
        return PROVIDERS[name]
    except KeyError:
        raise ValueError(f"LG_PROVIDER must be one of: {', '.join(PROVIDERS)}") from None


def close_model_providers() -> None:
    """Attempt all shared cleanups, including when an earlier provider fails."""
    from contextlib import ExitStack

    with ExitStack() as resources:
        for provider in PROVIDERS.values():
            resources.callback(provider.close)
