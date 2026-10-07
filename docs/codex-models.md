<!-- Copyright (c) 2026 Martin.Bechard@DevConsult.ca -->
# Codex request/response adapter

Set `LG_PROVIDER=codex` and an explicit `LG_MODEL` to run a text model through
the installed Codex CLI and its existing login. The adapter was verified with
Codex CLI 0.159.2. Check `codex --version` and `codex login status` first; use
`codex login` if needed. No new Python dependency or provider API key is required.
The model must be available to that Codex account. There is no automatic fallback.

```sh
LG_PROVIDER=codex LG_MODEL=gpt-6-sol LG_MAX_TOKENS= LG_EFFORT=medium \
  uv run python -m agent_runtime --sample simple_chat --client static --live \
  --env-file .env.local --prices models.json \
  --out reports/model_comparison/codex-gpt-6-sol
```

These settings can instead be saved in `.env.local`. `LG_CODEX_CLI` optionally
selects a different executable; on Windows use the native Codex executable.
For the minimal gateway profile, set
`LG_CODEX_MODEL_CATALOG=~/.codex/models_cache.json` to the existing Codex model
cache, or supply a catalog containing the exact selected model. The adapter
creates a temporary copy that removes native tool advertisement. It preserves
reasoning capabilities, routing, context limits, and all other model metadata;
it never edits the source cache. Missing or ambiguous model entries fail before
a CLI call. Leave this setting unset to retain the default model catalog.
`LG_AVAILABLE_MODELS` and symbolic model mappings apply as for other providers,
but an unavailable or disallowed model fails instead of falling back. Omit
`LG_EFFORT` to retain the CLI default. `--static` still selects the scripted model.
`LG_MAX_TOKENS` is unsupported and must be unset or blank.

`CodexChatModel` and `CopilotChatModel` share `GatewayChatModel` for history,
tracing, JSON tool decisions, and usage normalization. All providers, including
native API models, are constructed through the shared `ModelProvider` protocol;
see the [provider boundary](chat-composition.md#provider-boundary).
Each synchronous or asynchronous invocation starts `codex exec --json` with an
ephemeral session in a temporary directory. It supplies system instructions
through a temporary instruction file and role-labeled conversation history on
stdin. LangGraph owns history; the adapter never resumes a Codex conversation.
The requested model and effort flow into trace capture. CLI events identify the
selected model through the explicit command argument, not an independently
reported model identity.

Bound graph tool schemas are sent to the model with a shared JSON response
contract. The model returns either text or tool requests; the adapter converts
those decisions into LangChain messages. LangGraph validates tool arguments,
executes tools through its normal approval boundaries, and supplies the results
on the next request. Tool names and choice constraints are checked by the adapter;
malformed decisions fail visibly without guessing or falling back to plain text.
Without the minimal profile, tools with fixed-property argument schemas also receive a JSON
response schema through `--output-schema`, including the actual argument types.
The model supplies parameters explicitly, including declared defaults. Tools
with arbitrary-key objects or unresolved schema references use the prompted JSON
protocol because a closed response schema cannot represent them faithfully.
The minimal profile sends each complete application tool schema once, in compact
prompt JSON, and omits `--output-schema`. The same local decision validator and
graph argument validation still apply. Live tests with all three models verified
a tool call, default arguments, use of a unique returned observation, and a final
answer without repeating the completed call. This supports tool-dependent
workflows such as `context_budget`.

Native shell, apps, plugins, memory, browser, computer, image, skill-search, and
hook features remain disabled. The adapter skips user config and project
instruction discovery, suppresses automatic skill instructions, environment and
collaboration context, and disables native goal and synchronous question tools. Wire-level
inspection with CLI 0.159.2 confirmed zero native tool definitions with the
minimal catalog profile. The model metadata's `tool_mode`, `shell_type`,
`apply_patch_tool_type`, and `experimental_supported_tools` are overridden only
in the temporary copy. The profile also omits the textual sandbox briefing.
Global AGENTS.md guidance can still apply; `--ignore-user-config` does not remove
it. No MCP server tools appeared in the inspected requests. The CLI uses a read-only sandbox and denies approval
requests. Native non-text work invalidates the result: the CLI is only an LLM
gateway, never the workflow's tool executor. Multimodal content, stop sequences,
and arbitrary per-invocation overrides remain unsupported. It does not stream
responses. See the
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

## Timing measurements

Codex calls collect time to first token through OTLP/JSON on a short-lived
loopback receiver. The adapter passes the exporter explicitly because it uses
`--ignore-user-config`. `LG_CODEX_OTEL=false` disables collection for older CLI
versions. Construction and static runs do not start a receiver or call a model.

Only `codex.turn.ttft.duration_ms` is retained in each model step's sanitized
`codex_telemetry` receipt. Other metrics, prompts, log bodies, file paths,
credentials, and unrelated attributes are discarded. Complete turn time comes
from the existing client model-call span; output rate uses its reported output
tokens divided by elapsed time. These are the three comparison measurements.

Direct OpenAI API calls stream Responses output. The callback recorder measures
client time to first text, tool arguments, or exposed reasoning text, ignoring
empty lifecycle and usage chunks. Codex first-token timing comes from inside
the CLI and may include hidden reasoning; this differs from the API boundary. Complete turn time includes transport and startup for every provider.
Missing, malformed, or non-streaming first-token evidence stays unavailable.
Partial capture shows measured-call coverage. Test-input and judge calls are
excluded from the comparison. Token charges continue to use usage receipts.

Cumulative exports replace previous samples; delta exports retain unique
intervals. Multiple native turns cannot masquerade as one first-token duration.
The collector was exercised with Codex CLI 0.159.2.

See the [official telemetry documentation](https://learn.chatgpt.com/docs/config-file/config-advanced#observability-and-telemetry)
and [exporter configuration](https://learn.chatgpt.com/docs/config-file/config-reference).

## Request payload inspection

A loopback Responses receiver captured serialized requests for the same saved
simple-chat input and eight application tools. It rejected requests before
inference and retained no headers or credentials. Separate live calls measured
billed input tokens. Serialized character counts and billed tokens are different
measurements.

The initial GPT-5.5 request contained approximately 25 KB of skills context,
5.6 KB of native tool definitions, 12.4 KB of application instructions/tool
schemas, and 6.5 KB of structured-response configuration. The direct API sent
approximately 10.8 KB in total. No configured MCP server tools appeared.

The reductions were tried in this order:

1. Native configuration switches removed skills, environment context,
   delegation roles, and question/goal/clock tools. Skipping skill discovery alone
   was insufficient; prompt inclusion had to be disabled separately.
2. Model-catalog inspection explained why native tools survived the switches:
   Luna and Sol advertised code mode, and other metadata advertised apply_patch
   or experimental tools. A temporary copy of exact model metadata with these
   tool fields disabled removed every native tool without a CLI fork or proxy.
3. The application schemas were duplicated in the prompt and output schema.
   Schema-only experiments caused duplicate decisions or repeated completed tool
   calls. The adopted profile keeps the complete schemas once in the prompt,
   with strict local decision validation. Conversation instructions explicitly
   continue after the last message, including a completed tool observation.
4. Compact JSON and removal of redundant API function wrappers reduced prompt
   formatting overhead without deleting descriptions, parameters, or defaults.
5. Global identity guidance was another visible input block. Martin removed that
   personalization. An isolated runtime/login experiment became unnecessary and
   was not retained. The adapter continues using the ordinary existing login.

A shorter gateway instruction experiment also caused invalid commentary before
JSON. The retained instructions explicitly require one final JSON decision and
preserve the application tool boundary. Further removing tool contracts merely
to lower tokens would change the evaluated workflow.

Before replacing the DeepAgents factory, identical first-turn input with eight
application tools produced these measurements:

| Route/model | Input tokens | Serialized request characters | Native tools |
| --- | ---: | ---: | ---: |
| OpenAI API GPT-5.5 | 1,999 | 10,761 | 0 |
| OpenAI API GPT-5.6 Luna | 1,999 | — | 0 |
| Codex GPT-5.5 | 2,509 | 13,644 | 0 |
| Codex GPT-5.6 Luna | 2,513 | 13,900 | 0 |
| Codex GPT-6 Sol | 2,513 | 13,893 | 0 |

All routes still received the same eight application tool definitions. The
remaining 510–514-token difference includes the textual gateway contract and
Codex serialization; it is not an MCP catalog. The request inspection cannot
attribute every server-counted token to a local field. These matched first-turn
probes are separate from the adaptive two-turn sample comparisons below.

Both routes request medium effort with no caller-imposed output limit. Codex
still uses a textual conversation/tool protocol and its own request transport.
Its model profiles select low verbosity; Luna and Sol also set
`reasoning.context=all_turns`, whereas the direct API request only specifies
effort. Equal reasoning-token counts or response lengths are therefore not
guaranteed even after removing native tools and global guidance.

The [Codex TypeScript SDK](https://github.com/openai/codex/blob/main/sdk/typescript/README.md)
wraps the CLI. Switching to that SDK alone retains the Codex harness and does
not produce the direct OpenAI SDK's native Responses payload.

The upstream [request to disable built-in tools](https://github.com/openai/codex/issues/6049)
is still open. A [community minimal-prompt guide](https://github.com/abraxarion/How-to-shrink-the-system-prompt-for-Codex)
describes configuration and isolated-home approaches. The implementation here
uses native configuration and the model catalog, verified against actual request
bodies; it does not require a custom CLI build or request-rewriting proxy.

## Tool-free LangGraph comparison

The agent factory was then changed from `create_deep_agent` to the standard
LangGraph-backed `create_agent`. Simple chat now registers no tools, so neither
route receives the eight unused application definitions. The Codex gateway also
omits its JSON tool-decision protocol when no tools are bound. Other workflows
register only their explicit tools and selected middleware.

All five comparison columns were rerun with medium Agent effort, no API output
cap, and the same saved shared QA rubric:

| Route/model | First-turn input | Two-turn input | Agent cost, USD |
| --- | ---: | ---: | ---: |
| OpenAI API GPT-5.5 | 40 | 355 | 0.016835 |
| Codex GPT-5.5 | 89 | 447 | 0.014415 |
| OpenAI API GPT-5.6 Luna | 40 | 319 | 0.0005306 |
| Codex GPT-5.6 Luna | 93 | 350 | 0.0004612 |
| Codex GPT-6 Sol | 93 | 315 | 0.002590 |

Each completed both turns with complete first-token timing and scored 100 for
goal achievement and answer quality. The first turn has the same user/system
input; later turns have adaptive follow-ups and different answer lengths. The
remaining first-turn difference is 49–53 input tokens, with no application or
native tool definitions. Codex still receives JSON-encoded conversation history
and brief instructions explaining that encoding and prohibiting native actions.
The report's cost recap shows the saved counts and rates for every charge.
