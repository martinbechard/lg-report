"""Verify provider endpoint settings reach the real SDK without model requests.

Use dummy credentials and an isolated environment to prove .env configuration,
shell precedence, and the default public endpoint without accessing an account.
AI attribution: Generated with AI assistance by Northstar.
Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

import os
from unittest.mock import patch

import pytest
from dotenv import dotenv_values

from agent_runtime.harness.model_config import configured_model


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
