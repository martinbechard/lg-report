<!-- Copyright (c) 2026 Martin.Bechard@DevConsult.ca; third-party source excerpts retain their original rights. -->
# GitHub Copilot models through a local server

Select `LG_PROVIDER=copilot` to use GitHub Copilot through its local SDK server.
The server runs locally; model inference uses your GitHub Copilot account.
This adapter handles **text request/response with all tools disabled**. Choose
`simple_chat` for a first run. Tool-dependent samples cannot complete their tool
work with this provider.

## Installation and login

Install the optional Python dependency and its matching runtime:

```sh
uv sync --extra copilot --extra chat
uv run --extra copilot python -m copilot download-runtime
```

The SDK pins its compatible runtime and can download it automatically on first
use. The interactive CLI supplies the one-time login flow. To install it locally
in this checkout without changing your global npm installation:

```sh
npm install --prefix .cache/lg-report/copilot-cli @github/copilot
.cache/lg-report/copilot-cli/node_modules/.bin/copilot login
```

Existing Copilot login credentials are reused. Do not put credentials in Git.
The interactive CLI and SDK runtime have independent versions; the SDK uses its
matching runtime by default. `COPILOT_CLI_PATH` can override the executable when
exported in the shell, but it must be compatible with the pinned SDK.

## Configure and run

Add these settings to your existing `.env.local`, preserving unrelated settings:

```dotenv
LG_PROVIDER=copilot
LG_MODEL=gpt-5.4
# Optional: use a different local connection configuration path.
LG_COPILOT_CONFIG=.cache/lg-report/copilot.json
# Optional, model-dependent:
# LG_EFFORT=low
```

An explicit model code is required. `auto` is rejected. `LG_MODEL_ADVANCED` and
other symbolic model mappings still work. `LG_AVAILABLE_MODELS`, when supplied,
restricts the selected code; Copilot rejects a disallowed code instead of silently
switching to the default. The server's account model catalog is checked before
sending a prompt, and an unavailable model is an error.

Unset `LG_MAX_TOKENS`: this SDK session interface has no equivalent output-token
limit. Stop sequences, multimodal messages, tool results, and arbitrary invocation
overrides are also unsupported and raise errors. No OpenAI or Anthropic API key
is required. Selecting Copilot enables live mode; `--demo` still forces simulation.

```sh
uv run --extra copilot python -m agent_runtime --sample simple_chat --live --env-file .env.local
# Browser client:
uv run --extra copilot --extra chat python -m agent_runtime --sample simple_chat --client angular --live --env-file .env.local
```

Model construction is lazy. On the first request, the adapter checks the saved
server endpoint with an SDK handshake. If no usable server exists, it scans from
**7001 upward**, skips occupied ports belonging to other services, and starts on
the first free port. An existing compatible Copilot server encountered during the
scan can be reused. A stale saved port does not skip newly available lower ports.
Only after a successful handshake does it save the port and local connection
secret in `LG_COPILOT_CONFIG`. The default file is Git-ignored and created with
owner-only permissions. This is connection configuration, not report evidence.

The server stays available across requests in the application. Console and web
application shutdown stop an owned runtime; connections to an external server are
detached without stopping that server. Direct Python adapter users must call
`close_copilot_servers()` from `agent_runtime.harness.copilot_model` in their own
shutdown `finally` block.

## Request and tool behavior

Each invocation creates a fresh SDK session with the selected model, empty
`available_tools` and `tools` lists, and a deny-all permission handler. SDK
`mode="copilot-cli"` with `use_logged_in_user=True` uses the authenticated
Copilot SDK path. Authentication does not enable tools: session options separately
disable configuration discovery, custom instructions, skills, file hooks, host Git
operations, and the session store; no plugin, skill, or MCP configuration is supplied.
The supplied system message replaces Copilot's default instructions; conversation
history is passed as role-labelled JSON text. This is a text history encoding,
not a native multi-message model API. There is no streaming or automatic context
compaction. The temporary session is disconnected and deleted after each request,
including failures.

Graph tool bindings are intentionally suppressed, including DeepAgents' default
tools. A forced tool choice is rejected. The local graph still owns the workflow;
tool-disabled Copilot never executes its built-in file, shell, or MCP tools.

Reported usage counts are captured when supplied. Missing counts remain unknown.
Reports retain provider `copilot`. Missing model prices are retrieved from
[GitHub's Copilot pricing tables](https://docs.github.com/en/copilot/reference/copilot-billing/models-and-pricing),
saved in the daily cache, and included in the run's `prices.json`. Model codes
such as `claude-opus-5.5` keep Copilot's spelling and rates. Published long-context
rates apply above their input-token threshold in both HTML and Excel.

These dollar estimates describe token usage before plan allowances, not your
subscription invoice or legacy premium-request billing. Unavailable models,
changed page formats, and missing usage remain explicitly unknown. An explicit
`--prices` file still disables fetching and can supply `copilot:<model>` overrides.

## Model names and codes

The following codes are listed in GitHub's
[Copilot CLI model reference](https://docs.github.com/en/copilot/reference/copilot-cli-reference/cli-command-reference#supported-models),
checked on **2026-10-05**. This is a published catalog snapshot, not a guarantee
that every code is enabled for your account, plan, organization, or runtime.

| Model name | Code for `LG_MODEL` |
| --- | --- |
| Claude Sonnet 4.6 | `claude-sonnet-4.6` |
| Claude Opus 5.5 | `claude-opus-5.5` |
| Claude Haiku 4.5 | `claude-haiku-4.5` |
| GPT-5.4 | `gpt-5.4` |
| GPT-5.3-Codex | `gpt-5.3-codex` |
| GPT-6 Astra | `gpt-6-astra` |
| GPT-6 Sol | `gpt-6-sol` |
| GPT-6 Luna | `gpt-6-luna` |
| Gemini 3.7 Flash | `gemini-3.7-flash` |

List the codes and display names actually returned for your account:

```sh
uv run --extra copilot python -m agent_runtime.harness.copilot_model
```

This command starts/reuses the same local server without generating a model
answer. It reads `LG_COPILOT_CONFIG` from the shell (default shown above); it does
not load `.env.local`. An unauthenticated account, empty catalog, or unavailable
model must be resolved through Copilot login/account access, not by silently
falling back to another provider.

Upstream references: [Python SDK](https://github.com/github/copilot-sdk/blob/main/python/README.md),
[server setup](https://github.com/github/copilot-sdk/blob/main/docs/setup/backend-services.md),
and [model availability](https://docs.github.com/en/copilot/reference/ai-models/supported-models).
