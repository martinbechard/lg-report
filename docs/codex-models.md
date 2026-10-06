<!-- Copyright (c) 2026 Martin.Bechard@DevConsult.ca -->
# Codex request/response adapter

Set `LG_PROVIDER=codex` and an explicit `LG_MODEL` to run a text model through
the installed Codex CLI and its existing login. The adapter was verified with
Codex CLI 0.159.2. Check `codex --version` and `codex login status` first; use
`codex login` if needed. No new Python dependency or provider API key is required.
The model must be available to that Codex account. There is no automatic fallback.

```sh
LG_PROVIDER=codex LG_MODEL=gpt-6.1-sol LG_MAX_TOKENS= LG_EFFORT=low \
  uv run python -m agent_runtime --sample simple_chat --client static --live \
  --env-file .env.local --prices models.json \
  --out reports/model_comparison/codex-gpt-6.1-sol
```

These settings can instead be saved in `.env.local`. `LG_CODEX_CLI` optionally
selects a different executable; on Windows use the native Codex executable.
`LG_AVAILABLE_MODELS` and symbolic model mappings apply as for other providers,
but an unavailable or disallowed model fails instead of falling back. Omit
`LG_EFFORT` to retain the CLI default. `--demo` still selects the scripted model.
`LG_MAX_TOKENS` is unsupported and must be unset or blank.

`CodexChatModel` and `CopilotChatModel` share `TextOnlyChatModel` for history,
tracing, tool restrictions, and usage normalization. All providers, including
native API models, are constructed through the shared `ModelProvider` protocol;
see the [provider boundary](chat-composition.md#provider-boundary).
Each synchronous or asynchronous invocation starts `codex exec --json` with an
ephemeral session in a temporary directory. It supplies system instructions
through a temporary instruction file and role-labeled conversation history on
stdin. LangGraph owns history; the adapter never resumes a Codex conversation.
The requested model and effort flow into trace capture. CLI events identify the
selected model through the explicit command argument, not an independently
reported model identity.

The adapter suppresses graph tool bindings. Native shell, apps, plugins, memory,
browser, computer, image, skill-search, and hook features are disabled; user
configuration and project instructions are excluded from the request. The CLI
uses a read-only sandbox and denies approval requests. A native non-text work
event invalidates the result. This is for `simple_chat` and other text workflows;
tool-dependent samples, tool messages, multimodal content, stop sequences, and
per-invocation overrides are unsupported. It does not stream responses. See the
[Codex configuration reference](https://developers.openai.com/codex/config-reference/)
for the native feature and instruction settings.

Calls have a default 120-second timeout. The adapter stops and reaps its owned CLI
process on failure, timeout, or cancellation; it does not need a persistent server
or application shutdown hook. On POSIX it also terminates the process group,
including the Node launcher and its native child. CLI errors remain errors and
never become simulated successes. Error bodies and stderr are excluded from
exceptions to avoid exposing request payloads or credentials.

Input/output tokens, cached input, and reasoning output come from the CLI's
completed-turn receipt. Missing usage remains unknown. Provider identity stays
`codex`. Estimated costs use the corresponding model's OpenAI API token rates:
`codex:<model>` maps to `openai:<model>` in the saved `prices.json` aliases.
This is an API-equivalent usage estimate, not a subscription charge. Explicit
Codex tariffs or aliases override this default. Unknown model rates or missing
usage remain incomplete. Supplied pricing files stay offline; automatic discovery
fetches the exact underlying OpenAI model's tariff when necessary. CLI startup and
Codex harness context affect elapsed time
and token counts, even with the same user prompts as a direct API run. Compare
the responses and recorded evidence without treating this as a controlled latency
or cost benchmark.
