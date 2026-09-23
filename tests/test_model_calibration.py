"""Verify explicit calibration persistence without provider calls or credentials.

Exercise mixed success, offline report consumption, and strict source parsing
so a failed verification cannot silently reuse a guessed capacity.
AI attribution: Generated with AI assistance by Northstar.
Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from reporting.context import context_capacity
from reporting.pricing import Prices, load_prices
from scripts import calibrate_models as calibration


def test_calibration_checks_all_real_models_and_preserves_prices(monkeypatch):
    """Partial failure remains unknown while successful and demo lookups work offline."""
    data = json.loads(Path("models.json").read_text())
    original_models = json.loads(json.dumps(data["models"]))
    calls = []

    def verify(provider, model):
        """Fail Luna while proving later catalog entries are still checked."""
        calls.append(f"{provider}:{model}")
        if model == "gpt-5.6-luna":
            raise ValueError("secret must not be persisted")
        return 123456, "https://example.test/model", model

    monkeypatch.setattr(calibration, "verify_model", verify)
    assert calibration.calibrate(data) is False
    assert len(calls) == 6
    assert data["models"] == original_models
    assert "secret" not in json.dumps(data)
    prices = Prices.model_validate(data)
    assert context_capacity("openai", "gpt-5.6-luna", prices) is None
    assert context_capacity("demo", "scripted-chat", prices) is None
    assert context_capacity("openai", "gpt-5.6", prices)["capacity"] == 123456
    # Calibration survives the same serialization used for report snapshots.
    restored = Prices.model_validate_json(prices.model_dump_json())
    assert (
        context_capacity("anthropic", "claude-sonnet-5", restored)["capacity"] == 123456
    )


@pytest.mark.parametrize(
    "body,valid",
    [
        ("Model ID: `test-model`\n- 1,050,000 context window\n", True),
        ("Model ID: `wrong`\n- 1,050,000 context window\n", False),
        (
            "Model ID: `test-model`\n- 1,050,000 context window\n- 200,000 context window\n",
            False,
        ),
    ],
)
def test_openai_requires_exact_unambiguous_source(monkeypatch, body, valid):
    """Model access alone cannot establish a context capacity."""
    client = MagicMock()
    client.__enter__.return_value.models.retrieve.return_value.id = "test-model"
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(calibration, "OpenAI", lambda **kwargs: client)
    monkeypatch.setattr(calibration, "fetch_text", lambda url: body)
    if valid:
        assert calibration.verify_model("openai", "test-model")[0] == 1050000
    else:
        with pytest.raises(ValueError):
            calibration.verify_model("openai", "test-model")


def test_anthropic_missing_capacity_is_unknown(monkeypatch):
    """An accessible model without capacity metadata must not inherit a guess."""
    client = MagicMock()
    client.__enter__.return_value.models.retrieve.return_value = SimpleNamespace(
        id="test-model",
        max_input_tokens=None,
    )
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(calibration, "Anthropic", lambda **kwargs: client)
    with pytest.raises(ValueError):
        calibration.verify_model("anthropic", "test-model")


def test_cli_writes_config_only_when_invoked(monkeypatch, tmp_path):
    """A complete explicit pass publishes validated JSON with original tariffs."""
    config = tmp_path / "models.json"
    config.write_text(Path("models.json").read_text())
    monkeypatch.setattr("sys.argv", ["calibrate_models.py", "--config", str(config)])
    monkeypatch.setattr(calibration, "load_dotenv", lambda: None)
    monkeypatch.setattr(
        calibration, "verify_model", lambda p, m: (10000, "metadata", m)
    )
    assert calibration.main() == 0
    assert len(load_prices(config).calibrations) == 6
    assert list(tmp_path.iterdir()) == [config]
