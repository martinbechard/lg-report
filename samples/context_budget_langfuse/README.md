<!-- Copyright (c) 2026 Martin.Bechard@DevConsult.ca -->
# Watch a coding team in Langfuse

Build a small Python function that turns a title such as ` Hello World! ` into
a URL-friendly slug such as `hello-world`. A planner writes a file plan, a
worker writes the code and tests, and an independent reviewer checks the files
and test results. Review feedback can send the worker back to fix its work.

This is the existing [context-budget coding workflow](../context_budget/README.md)
with Langfuse recording enabled. It creates real files and executes generated
Python tests. The local HTML report and Langfuse observe the same execution.

## Run with real agents

Start Langfuse at [localhost:3001](http://localhost:3001). The root `.env.local`
must contain `LANGFUSE_BASE_URL`, `LANGFUSE_PUBLIC_KEY`, and `LANGFUSE_SECRET_KEY`.
Run from the repository root. This example uses an existing Codex login and a
short request, avoiding the repeated text in the context-stress fixture:

```sh
LG_PROVIDER=codex LG_MODEL=gpt-6-sol LG_MAX_TOKENS= LG_EFFORT=medium \
uv run python -m agent_runtime --sample context_budget_langfuse \
  --env-file .env.local --live --client static --no-qa \
  --request 'Build slugify(title) in slug.py using only the Python standard library. Trim whitespace, lowercase, replace runs of non-alphanumeric characters with a hyphen, and strip boundary hyphens. Write test_slug.py with unittest tests for ordinary titles, empty input, and repeated punctuation. Make a file plan, run the tests, and obtain an independent review.' \
  --out .local-reports/coding-langfuse \
  --option 'workspace_dir=".local-reports/coding-langfuse/workspace"'
```

Here `--live` means real agents. `--client static` supplies the written request
without asking another model to play the user; it does not simulate the agents.
`--no-qa` disables the extra report-scoring judge, not the workflow's code reviewer.
Provider/model overrides apply only to this command. Other configured providers
also work. API providers require valid credentials and may charge for calls.

For an interactive conversation, replace `--client static` with `--client console`
and omit `--request`. Ask for an improvement after the first answer to add another
turn trace to the same session. Each fresh launch creates a new session.

## Follow the work

1. Open the **Session** URL printed when the run finishes, then open a turn trace.
2. In **Tree**, expand the planner, worker, and isolated-reviewer branches.
   Inspect file-tool inputs/outputs to see the plan, source, and tests.
3. Find the test execution step and inspect its receipt: test count, exit status,
   and output. A model saying “tests passed” is not execution evidence.
4. Follow any review rejection back to the worker's repair. Completion should
   follow approval; a repair-limit result may be incomplete and is not success.
5. Switch to **Timeline** to compare call durations. Select model generations
   for prompts, replies, and token usage. Missing prices remain unknown.

The retained workspace contains `plan.md`, `slug.py`, and `test_slug.py` when
created. Open `.local-reports/coding-langfuse/report.html` for the local report.
Use a fresh output/workspace path for a separate exercise. These local outputs
are not stable checked-in samples. Without `workspace_dir`, files are temporary.

## Boundaries and repeatable teaching run

Only the three workflow files are exposed through the agent file backend.
The reviewer can read but cannot edit; worker writes are restricted to its
assignment. There is no human approval gate. Generated tests execute locally
with a timeout and without provider credentials; this is not an OS sandbox.
Use this dev exercise with trusted requests. See the base workflow README for
its execution and review limits.

For a repeatable walkthrough without model calls:

```sh
uv run python -m agent_runtime --sample context_budget_langfuse \
  --env-file .env.local --static --no-qa --out .local-reports/coding-langfuse-scripted
```

That mode uses scripted model decisions but real files, test execution, and
Langfuse ingestion. Its deliberately long prompts exercise context compaction;
the live command above uses a concise practical request. Traces stay private by
default. Langfuse records submitted content, so avoid putting secrets in prompts
or exercise files.
