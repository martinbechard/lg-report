"""Explicitly verify configured model access and published context capacity.

Update only calibration evidence in the selected models.json. Normal execution
and rendering never invoke this command. Credentials come from the environment
or .env and are never saved. This checks metadata, not large-prompt acceptance;
no generation requests are made and no pricing fields are changed.
AI attribution: Generated with AI assistance by Northstar.
Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

import argparse
import json
import os
import re
from datetime import UTC, datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.parse import quote

from anthropic import Anthropic
from anthropic import APIError as AnthropicAPIError
from dotenv import load_dotenv
from openai import APIError as OpenAIAPIError
from openai import OpenAI

from reporting.price_refresh import fetch_text
from reporting.pricing import ModelCalibration, Prices


def verify_model(provider: str, model: str) -> tuple[int, str, str]:
    """Verify access with the provider key and return capacity plus its source.

    SDK retries are disabled to keep an explicit pass bounded. OpenAI retrieval
    proves model visibility, not inference permission; its official page supplies
    capacity. Anthropic supplies capacity directly. Never infer limits from names
    or bundled LangChain profiles, which could repeat the stale lookup problem.
    """
    if provider == "openai":
        with OpenAI(
            api_key=os.environ["OPENAI_API_KEY"], timeout=20, max_retries=0
        ) as client:
            resolved = client.models.retrieve(model).id
        url = (
            f"https://developers.openai.com/api/docs/models/{quote(model, safe='')}.md"
        )
        body = fetch_text(url)
        if f"Model ID: `{model}`" not in body:
            raise ValueError("Official page identity does not match")
        matches = re.findall(r"(?m)^- ([\d,]+) context window\s*$", body)
        if len(matches) != 1:
            raise ValueError("Missing or ambiguous official context window")
        capacity = int(matches[0].replace(",", ""))
    elif provider == "anthropic":
        with Anthropic(
            api_key=os.environ["ANTHROPIC_API_KEY"], timeout=20, max_retries=0
        ) as client:
            info = client.models.retrieve(model)
        resolved = info.id
        capacity = info.max_input_tokens
        url = f"https://api.anthropic.com/v1/models/{quote(model, safe='')}"
    else:
        raise ValueError("Unsupported provider")
    if type(capacity) is not int or capacity <= 0:
        raise ValueError("Provider did not supply a positive context capacity")
    return capacity, url, resolved


def calibrate(data: dict) -> bool:
    """Check every real catalog entry, continuing after individual failures.

    Modify only the calibration section. Demo models inherit their declared
    basis at display time. Failure replaces old calibration with explicit unknown
    capacity; raw exceptions are excluded because they can contain credentials
    or response bodies. Return whether every real model passed.
    """
    prices = Prices.model_validate(data)
    checks = {}
    success = True
    for key in prices.models:
        provider, model = key.split(":", 1)
        if provider == "demo":
            print(f"{key}: illustrative; uses configured basis")
            continue
        checked = ModelCalibration(checked_at=datetime.now(UTC))
        try:
            checked.capacity, checked.source, checked.resolved_model = verify_model(
                provider, model
            )
            print(f"{key}: {checked.capacity:,} tokens ({checked.source})")
        except (
            OpenAIAPIError,
            AnthropicAPIError,
            OSError,
            ValueError,
            KeyError,
        ) as exc:
            # This per-model boundary intentionally catches SDK/network/parser
            # errors so one failure cannot prevent checks of the remaining models.
            checked.error = (
                f"Verification failed ({type(exc).__name__}); capacity unavailable."
            )
            print(f"{key}: {checked.error}")
            success = False
        checks[key] = checked.model_dump(mode="json")
    data["calibrations"] = checks
    Prices.model_validate(data)
    return success


def main() -> int:
    """Run only on explicit invocation and atomically publish the complete pass.

    Environment values override .env. A nonzero result means one or more checks
    failed, but their unknown statuses are still saved alongside successes.
    Invalid configuration fails before any provider requests or file replacement.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("models.json"))
    args = parser.parse_args()
    load_dotenv()
    data = json.loads(args.config.read_text(encoding="utf-8"))
    success = calibrate(data)
    temporary = None
    try:
        with NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=args.config.parent, delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(json.dumps(data, indent=2) + "\n")
        temporary.replace(args.config)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
