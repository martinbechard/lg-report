"""Verify provider settings and model restrictions without model requests.

Use dummy credentials and an isolated environment to prove .env configuration,
shell precedence, and the default public endpoint without accessing an account.
AI attribution: Generated with AI assistance by Northstar.
Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

import os
from unittest.mock import patch

import pytest
from dotenv import dotenv_values

from agent_runtime.harness import model_config
from agent_runtime.harness.model_config import configured_identity, configured_model


@pytest.fixture
def isolated_models():
    """Keep workstation restrictions and prior warning history out of each check."""
    with (
        patch.dict(os.environ, {}, clear=True),
        patch.object(model_config, "_warned_models", set()),
    ):
        yield


@pytest.mark.parametrize("allowlist", [None, "", "  ", " chosen , other "])
def test_allowed_or_unrestricted_models(isolated_models, caplog, allowlist):
    """Optional restrictions preserve explicit selection and emit no warning."""
    assert configured_identity(
        "chosen", settings={"LG_AVAILABLE_MODELS": allowlist}
    ) == ("openai", "chosen")
    assert not caplog.records


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
def test_fallback_warns_once_per_missing_model(isolated_models, caplog, provider):
    """Repeated identity/adapter resolution does not repeat a missing-model warning."""
    settings = {
        "LG_PROVIDER": provider,
        "LG_AVAILABLE_MODELS": " allowed , other ",
        "LG_MODEL": "allowed",
    }
    for requested in ["missing-a", "missing-a", "missing-b", "missing-b"]:
        assert configured_identity(requested, settings=settings) == (
            provider,
            "allowed",
        )
    assert len(caplog.records) == 2
    assert all("Using fallback LG_MODEL='allowed'" in r.message for r in caplog.records)


@pytest.mark.parametrize("fallback", [None, "", "  ", "disallowed"])
def test_unusable_fallback_fails(isolated_models, caplog, fallback):
    """A rejected request cannot use a built-in or explicitly disallowed default."""
    settings = {"LG_AVAILABLE_MODELS": "allowed", "LG_MODEL": fallback}
    for _ in range(2):
        with pytest.raises(ValueError, match="LG_AVAILABLE_MODELS"):
            configured_identity("missing", settings=settings)
    assert len(caplog.records) == 1
    assert "fallback" in caplog.text.lower()


def test_default_selection_is_restricted(isolated_models):
    """Omitting an explicit request must not bypass the same allowlist."""
    with pytest.raises(ValueError, match="No fallback is configured"):
        configured_identity(settings={"LG_AVAILABLE_MODELS": "allowed"})
    assert configured_identity(
        settings={"LG_AVAILABLE_MODELS": "allowed", "LG_MODEL": "allowed"}
    ) == ("openai", "allowed")


def test_local_file_fallback_reaches_adapter_and_identity(
    isolated_models, tmp_path, monkeypatch, caplog
):
    """File restrictions and shell overrides resolve before constructing the SDK."""
    env_file = tmp_path / ".env.local"
    env_file.write_text(
        "LG_AVAILABLE_MODELS=file-model\nLG_MODEL=file-model\n"
        "OPENAI_API_KEY=placeholder-not-a-real-key\n"
    )
    monkeypatch.setenv("LG_AVAILABLE_MODELS", "shell-model")
    monkeypatch.setenv("LG_MODEL", "shell-model")
    adapter, provider, model = configured_model(
        "missing", settings=dotenv_values(env_file)
    )
    assert adapter.model_name == model == "shell-model"
    assert provider == "openai"
    assert "Using fallback LG_MODEL='shell-model'" in caplog.text


@pytest.mark.parametrize(
    ("file_url", "shell_url", "expected"),
    [
        (
            "https://file-resource.openai.azure.com/openai/v1/",
            None,
            "https://file-resource.openai.azure.com/openai/v1/",
        ),
        (
            "https://file-resource.openai.azure.com/openai/v1/",
            "https://shell-resource.openai.azure.com/openai/v1/",
            "https://shell-resource.openai.azure.com/openai/v1/",
        ),
        (None, None, "https://api.openai.com/v1/"),
    ],
)
def test_openai_endpoint_from_env_file(tmp_path, file_url, shell_url, expected):
    """The app's file settings must configure both sync and async SDK clients."""
    env_file = tmp_path / ".env"
    contents = (
        "LG_PROVIDER=openai\nLG_MODEL=my-gpt-4.1-deployment\n"
        "OPENAI_API_KEY=placeholder-not-a-real-key\n"
    )
    if file_url:
        contents += f"OPENAI_BASE_URL={file_url}\n"
    env_file.write_text(contents)
    environment = {"OPENAI_BASE_URL": shell_url} if shell_url else {}
    # No real credentials or workstation endpoint may influence this check.
    with patch.dict(os.environ, environment, clear=True):
        adapter, provider, model = configured_model(settings=dotenv_values(env_file))
        assert str(adapter.root_client.base_url) == expected
        assert str(adapter.root_async_client.base_url) == expected
        assert adapter.use_responses_api is True
        assert (provider, model) == ("openai", "my-gpt-4.1-deployment")
        assert dict(os.environ) == environment
