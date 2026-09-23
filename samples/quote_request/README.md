<!-- Copyright (c) 2026 Martin.Bechard@DevConsult.ca -->
# Model-directed human clarification

The LLM decides when a request does not make sense, explains its concern, and
asks the human for clarification. There is no required-field checklist driving
questions. The initial fixture has contact information, service, quantity, and
delivery details, but multiple issues:

- Individually addressed invitations conflict with identical text and no variable printing.
- 500 invitations conflict with 650 guests each receiving their own invitation.
- If personalization belongs on envelopes, envelope printing and the address list need clarification.

The model considers the whole request after every answer. It can ask more
questions, revisit an unclear answer, group related questions, or finish when the
scope is sufficiently clear. Live questions and their order are model decisions;
recognition of every issue is not guaranteed by deterministic business rules.

Run a real conversation with your configured provider:

```sh
cp samples/quote_request/.env.example samples/quote_request/.env
# Set your provider credentials in the local .env file.
uv run python -m agent_runtime --sample quote_request --live --client console
```

The console displays the model's reason and question. Answer in ordinary language.
Type `/cancel` to abandon; EOF and Ctrl-C also cancel. The initial request and all
answers go to the configured model. Nothing is submitted to a supplier.

Use `--values /path/to/quote.json` with live mode to supply a free-form JSON object.
For example, a `description` string can contain the whole request. A coherent
request can finish without questions. There are no mandatory contact fields.

An offline demonstration explicitly replays authored model decisions and answers:

```sh
uv run python -m agent_runtime --sample quote_request --demo
uv run python -m agent_runtime --sample quote_request --demo --scenario cancel
```

This simulation demonstrates interrupt/resume wiring, multiple issues, and
cancellation; it is not evidence of LLM reasoning. Custom values and console
answers require live mode because scripted decisions cannot interpret them.

The graph follows assess → ask human → assess, or assess → complete. The
`quote_interpreter` agent owns the system prompt, message formatting, decision-tool
binding, model call, and output validation. The workflow supplies structured
request/conversation data and owns history retention, routing, checkpoints,
and interrupts. The model
returns a `QuoteDecision` tool call containing its action, explanation, and
question or summary. This output schema validates the model protocol, not the
business meaning of the quote. The agent returns a validated decision to the
workflow; the workflow never inspects model tool names or arguments. A separate
ask node uses a real LangGraph
interrupt, so resuming does not repeat the preceding model call. Human answers
are preserved verbatim and included in the next assessment.

Completion returns `status: completed` and a local `request.summary` of the agreed
scope. It does not guarantee feasibility or supplier acceptance. Cancellation
returns `status: cancelled` and clears current draft/conversation state, but does
not erase prior checkpoints or traces. Checkpoints last only for this process.

One shared recorder captures the session in HTML and independent `run.json`.
`--out` selects another reusable output directory; `--metadata-only` omits report payloads
but does not prevent sending the request to the model. `--prices models.json` and
`--fx-file /path/to/fx.json` select saved model-price and FX references; FX never performs a lookup here. Live usage comes from provider
callbacks; absent usage remains explicit. Offline decisions invent no usage.

Local reports default to `report.html`, `run.json`, `spans.jsonl`, and
`prices.json` in `reports/quote_request/`. The next default run replaces
these files. Use `--out reports/saved-run` to choose another reusable report directory.

## Angular client

Select **Quote clarification** in the catalog, or launch this sample with
`--client angular`. Questions remain paused until answered or cancelled. The
console and browser submit the same AG-UI resume contract to LangGraphAgent.
