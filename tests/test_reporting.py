"""Verify trace capture, normalized evidence, and reporting on real local graphs.

Scripted model responses avoid provider calls; fixed tariffs keep arithmetic
assertions stable. Tests include failed and interrupted execution because missing
usage or unfinished spans must not be reported as free successful work.

AI attribution: Generated with AI assistance.

Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

import json
from decimal import Decimal
from pathlib import Path
from uuid import uuid4

import pytest
from deepagents import create_deep_agent
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from lg_report.agents.chat_agent import build_agent as build_chat_agent
from lg_report.platform.simulated_model import ScriptedChatModel
from lg_report.report.capture import TraceCapture
from lg_report.report.normalize import normalize
from lg_report.report.pricing import cost, load_prices, summarize
from lg_report.report.recording import record_run
from lg_report.report.render import render
from lg_report.report.schema import Run, Step, Usage
from samples.simple_chat.test_case import make_simulated_model as make_chat_model


@pytest.fixture
def prices():
    # Keep report arithmetic tied to the repository fixture so provider
    # tariff changes cannot alter regression expectations.
    return load_prices(Path(__file__).parent / "fixtures/accounting_prices.json")


# Centralize run parsing so every report test validates the persisted
# schema rather than relying on an in-memory implementation detail.
def read_run(directory):
    # Return the validated Run from the output directory supplied by the test.
    # Reading run.json checks what a later report consumer actually receives.
    return Run.model_validate_json((directory / "run.json").read_text())


# Run the production Deep Agent path with a scripted model to prove
# normalized spans, privacy filtering, pricing, and HTML output agree.
def test_real_deepagents_offline_pipeline(tmp_path, prices):
    out = tmp_path / "run"
    record_run(
        build_chat_agent(make_chat_model()),
        {"messages": [("user", "secret input")]},
        out,
        prices,
        provider="demo",
        model="scripted-chat",
        demo=True,
    )
    run = read_run(out)
    summary = summarize(run, prices)
    assert run.status == "ok" and run.demo and run.output is None
    assert summary["model_calls"] == 1
    model = next(s for s in run.steps if s.kind == "model")
    assert summary["input_tokens"] == sum(
        value for key, value in model.context.items() if key.startswith("simulated_")
    )
    assert summary["output_tokens"] > 0
    assert model.usage.cache_read == 0
    assert (
        summary["known_cost"]
        == (
            Decimal(summary["input_tokens"]) * Decimal("0.4")
            + Decimal(summary["output_tokens"]) * Decimal("1.6")
        )
        / 1_000_000
    )
    assert summary["unpriced_calls"] == 0
    assert "secret input" not in (out / "spans.jsonl").read_text()
    assert "Offline demonstration" in (out / "report.html").read_text()
    spans = [
        json.loads(line) for line in (out / "spans.jsonl").read_text().splitlines()
    ]
    assert len({s["context"]["trace_id"] for s in spans}) == 1
    assert any(s["parent_id"] for s in spans)
    assert normalize(out / "spans.jsonl", title=run.title, demo=True) == run
    assert load_prices(out / "prices.json") == prices


# Construct the smallest valid model span for focused pricing and
# schema tests, leaving unrelated capture machinery out of those cases.
def model_step(**kwargs):
    # Return a model span with fixed identity/timing and caller-supplied fields.
    # kwargs adds optional schema fields such as usage, not duplicate base fields.
    return Step(
        id="m",
        name="model",
        kind="model",
        start_ns=1,
        end_ns=2,
        status="ok",
        provider="demo",
        model="scripted-chat",
        **kwargs,
    )


# Cache reads are an input subset and reasoning is an output subset; an
# unknown cache-write tariff must make the total explicitly unpriced.
def test_cache_and_reasoning_are_subsets(prices):
    step = model_step(
        usage=Usage(input_tokens=1000, output_tokens=100, cache_read=500, reasoning=50)
    )
    assert cost(step, prices)[0] == Decimal("0.00041")
    step.usage.cache_write = 100
    assert cost(step, prices)[0] is None
    prices.models["demo:scripted-chat"].cache_write = Decimal("0.5")
    assert cost(step, prices)[0] == Decimal("0.00042")


# Missing usage or model pricing is uncertainty, not zero cost; only
# an explicit zero-token usage record may produce a zero subtotal.
def test_unknown_tokens_and_model_are_not_free(prices):
    step = model_step()
    assert cost(step, prices)[0] is None
    step.usage = Usage(input_tokens=0, output_tokens=0)
    assert cost(step, prices)[0] == 0
    step.model = "unlisted"
    assert cost(step, prices)[0] is None


# Reject impossible token relationships and self-parenting traces at
# schema validation so downstream reports cannot normalize invalid structure.
def test_schema_rejects_invalid_counts_and_cycles():
    with pytest.raises(ValueError):
        Usage(input_tokens=10, output_tokens=1, cache_read=11)
    with pytest.raises(ValueError):
        Usage(input_tokens=-1, output_tokens=0)
    step = model_step()
    step.parent_id = step.id
    with pytest.raises(ValueError):
        Run(id="r", title="bad", status="ok", steps=[step])


# User and model content is untrusted HTML input; escaping must hold
# for titles, step names, and output while preserving the report shell.
def test_html_escapes_all_content(tmp_path, prices):
    step = model_step()
    step.name = '<img src=x onerror="alert(1)">'
    run = Run(
        id="r",
        title="<script>alert(1)</script>",
        status="ok",
        steps=[step],
        output=step.name,
    )
    output = tmp_path / "report.html"
    render(run, prices, output)
    html = output.read_text()
    assert "<script>alert(1)</script>" not in html and "<img " not in html
    assert html.count("<script>") == 1
    assert "&lt;script&gt;" in html and "&lt;img" in html
    assert "could not be priced" in html


# A real model-tool-model graph protects ordering, tool schema capture,
# and per-call arithmetic in the report pipeline.
def test_model_tool_model_sequence(tmp_path, prices):
    @tool(parse_docstring=True)
    def lookup(topic: str) -> str:
        # Provide a real executable tool for the model-tool-model recording test.
        # The docstring below is parsed into the tool schema and must stay stable.
        # The return is the executed tool result; AIMessage.tool_calls only requests it.
        """Look up a fact in a tiny fixture corpus.

        Args:
            topic: The topic used to label the fixed graph explanation.
        """
        return f"{topic}: a graph executes connected nodes."

    usage = {"input_tokens": 100, "output_tokens": 10, "total_tokens": 110}
    model = ScriptedChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "lookup", "args": {"topic": "LangGraph"}, "id": "call-1"}
                ],
                usage_metadata=usage,
            ),
            AIMessage(
                content="A graph executes connected nodes.", usage_metadata=usage
            ),
        ]
    )
    agent = create_deep_agent(model=model, tools=[lookup], subagents=[])
    out = tmp_path / "tool"
    record_run(
        agent,
        {"messages": [("user", "Look up LangGraph")]},
        out,
        prices,
        provider="demo",
        model="scripted-chat",
        include_output=True,
    )
    run = read_run(out)
    model_steps = [step for step in run.steps if step.kind == "model"]
    for step in model_steps:
        lookup_schema = next(
            item["function"]
            for item in step.tool_definitions
            if item.get("function", {}).get("name") == "lookup"
        )
        assert lookup_schema["parameters"]["properties"]["topic"]["description"]
    html = (out / "report.html").read_text()
    assert '<details class="tool-definitions">' in html
    assert "The topic used to label the fixed graph explanation." in html
    assert [s.kind for s in run.steps if s.kind != "workflow"] == [
        "model",
        "tool",
        "model",
    ]
    assert summarize(run, prices)["input_tokens"] == 200
    assert summarize(run, prices)["known_cost"] == Decimal("0.000112")


# Runtime failure must preserve a report and sanitized trace evidence
# so diagnosis remains possible without leaking exception content.
def test_failure_preserves_partial_report(tmp_path, prices):
    def fail(state):
        # Exercise failure reporting from inside a graph node.
        # LangGraph supplies state, but its contents do not affect the deliberate error.
        raise RuntimeError("secret exception content")

    graph = StateGraph(dict)
    graph.add_node("failing-node", fail)
    graph.add_edge(START, "failing-node")
    graph.add_edge("failing-node", END)
    out = tmp_path / "failure"
    with pytest.raises(RuntimeError, match="secret exception"):
        record_run(
            graph.compile(), {}, out, prices, provider="demo", model="scripted-chat"
        )
    run = read_run(out)
    assert run.status == "error"
    assert all(s.status == "error" for s in run.steps)
    assert "secret exception" not in (out / "spans.jsonl").read_text()
    assert (out / "report.html").exists()


# Closing an unfinished callback must produce an incomplete run state,
# preventing an abandoned span from being presented as successful work.
def test_unfinished_callback_is_marked_incomplete(tmp_path):
    path = tmp_path / "spans.jsonl"
    capture = TraceCapture(path, "demo", "scripted-chat")
    capture.on_chain_start({}, {}, run_id=uuid4())
    capture.close()
    run = normalize(path, title="Incomplete")
    assert run.status == "incomplete"


# Interrupt and resume are separate accounting invocations; both must
# retain truthful statuses around the approval boundary.
def test_interrupt_and_resume_separate_invocations(tmp_path, prices):
    def approval(state):
        # Create an approval boundary so pause and completion get separate reports.
        # On the first call interrupt exits this node through GraphInterrupt; the
        # graph handles it and returns the pending question. Command(resume=True)
        # re-enters this node from its beginning on the same checkpoint thread.
        # The repeated interrupt returns True, allowing the partial state update
        # below to reach the graph. No suspended Python stack is restored.
        return {"approved": interrupt("Approve this action?")}

    graph = StateGraph(dict)
    graph.add_node("approval", approval)
    graph.add_edge(START, "approval")
    graph.add_edge("approval", END)
    # This saver retains checkpoints only in this process. Both record_run
    # calls below use the same compiled graph and thread id to resume its pause.
    agent = graph.compile(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "test"}}
    record_run(
        agent,
        {},
        tmp_path / "pause",
        prices,
        provider="demo",
        model="scripted-chat",
        config=config,
    )
    assert read_run(tmp_path / "pause").status == "interrupted"
    record_run(
        agent,
        Command(resume=True),
        tmp_path / "resume",
        prices,
        provider="demo",
        model="scripted-chat",
        config=config,
    )
    assert read_run(tmp_path / "resume").status == "ok"


# Existing output is protected from accidental overwrite so reruns do
# not destroy prior evidence or mix two accounting sessions.
def test_existing_directory_is_not_overwritten(tmp_path, prices):
    with pytest.raises(FileExistsError):
        record_run(
            build_chat_agent(make_chat_model()),
            {},
            tmp_path,
            prices,
            provider="demo",
            model="scripted-chat",
        )


# Anthropic cache lifetime fields map to the correct pricing buckets;
# this catches provider-specific usage regressions without a live API.
def test_anthropic_cache_lifetime_usage(tmp_path, prices):
    from langchain_core.outputs import ChatGeneration, LLMResult

    capture = TraceCapture(tmp_path / "spans.jsonl", "anthropic", "claude-sonnet-4-6")
    run_id = uuid4()
    capture.on_chat_model_start({}, [], run_id=run_id)
    message = AIMessage(
        content="ok",
        usage_metadata={
            "input_tokens": 1000,
            "output_tokens": 100,
            "total_tokens": 1100,
            "input_token_details": {
                "cache_read": 100,
                "cache_creation": 0,
                "ephemeral_5m_input_tokens": 200,
                "ephemeral_1h_input_tokens": 300,
            },
        },
    )
    capture.on_llm_end(
        LLMResult(generations=[[ChatGeneration(message=message)]]), run_id=run_id
    )
    capture.close()
    run = normalize(tmp_path / "spans.jsonl", title="Cache test")
    assert run.steps[0].usage.cache_write == 500
    assert cost(run.steps[0], prices)[0] == Decimal("0.00528")


# Retry attempts are separate model spans and both remain visible so
# cost and failure history cannot be collapsed into a misleading success.
def test_retry_keeps_both_attempts(tmp_path, prices):
    from langgraph.types import RetryPolicy

    class FailOnce(ScriptedChatModel):
        attempts: int = 0

        def _generate(self, *args, **kwargs):
            # Make the first model attempt fail so retry accounting retains both spans.
            # Forward LangChain generation arguments unchanged on subsequent attempts;
            # the successful return is the parent scripted model generation result.
            self.attempts += 1
            # Fail only the initial invocation to exercise an actual graph
            # retry. Later attempts delegate to the model so the trace must
            # retain both the failed attempt and the successful token usage.
            if self.attempts == 1:
                raise RuntimeError("transient")
            return super()._generate(*args, **kwargs)

    model = FailOnce(
        responses=[
            AIMessage(
                content="ok",
                usage_metadata={
                    "input_tokens": 20,
                    "output_tokens": 5,
                    "total_tokens": 25,
                },
            )
        ]
    )
    graph = StateGraph(dict)
    graph.add_node(
        "model",
        # This node ignores its graph state: each attempt invokes the same
        # model with a fixed prompt and returns its AIMessage as an answer
        # state update. RetryPolicy reruns the node after RuntimeError.
        lambda state: {"answer": model.invoke("hi")},
        retry_policy=RetryPolicy(
            max_attempts=2, initial_interval=0.001, jitter=False, retry_on=RuntimeError
        ),
    )
    graph.add_edge(START, "model")
    graph.add_edge("model", END)
    record_run(
        graph.compile(),
        {},
        tmp_path / "retry",
        prices,
        provider="demo",
        model="scripted-chat",
    )
    run = read_run(tmp_path / "retry")
    model_steps = [s for s in run.steps if s.kind == "model"]
    assert [s.status for s in model_steps] == ["error", "ok"]
    assert summarize(run, prices)["input_tokens"] == 20
    assert summarize(run, prices)["missing_usage"] == 1


# A handled tool error is part of conversation evidence and must
# remain visible even when the graph continues to a final response.
def test_handled_tool_error_is_visible(tmp_path):
    from langchain_core.messages import ToolMessage

    path = tmp_path / "spans.jsonl"
    capture = TraceCapture(path, "demo", "scripted-chat")
    run_id = uuid4()
    capture.on_tool_start({"name": "lookup"}, "", run_id=run_id)
    capture.on_tool_end(
        ToolMessage(content="sensitive failure", tool_call_id="call", status="error"),
        run_id=run_id,
    )
    capture.close()
    run = normalize(path, title="Handled error")
    assert run.steps[0].status == "error"
    assert "sensitive failure" not in path.read_text()


# Failure before callback initialization must still surface the real
# exception and leave a truthful failed report.
def test_failure_before_callbacks_is_not_masked(tmp_path, prices):
    class BrokenAgent:
        def invoke(self, inputs, config):
            # Exercise failure before an agent has emitted any tracing callbacks.
            # Accept the recorder inputs/config contract but raise before using either.
            raise RuntimeError("original failure")

    out = tmp_path / "early-failure"
    with pytest.raises(RuntimeError, match="original failure"):
        record_run(
            BrokenAgent(), {}, out, prices, provider="demo", model="scripted-chat"
        )
    assert read_run(out).status == "error"
    assert (out / "report.html").exists()


@pytest.mark.parametrize("capture_content", [False, True])
@pytest.mark.parametrize("provider", ["openai", "anthropic"])
def test_tool_definition_capture_and_render(
    tmp_path, prices, capture_content, provider
):
    """Preserve provider schemas only with content capture and escape their text."""
    definition = {
        "name": "unused_tool",
        "description": "Private description <script>alert(1)</script>",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Search phrase"}},
            "required": ["query"],
        },
    }
    if provider == "openai":
        definitions = [{"type": "function", "function": definition}]
    else:
        definition["input_schema"] = definition.pop("parameters")
        definitions = [definition]
    path = tmp_path / "spans.jsonl"
    capture = TraceCapture(path, provider, "fixture", capture_content=capture_content)
    run_id = uuid4()
    capture.on_chat_model_start(
        {},
        [[]],
        run_id=run_id,
        invocation_params={"tools": definitions, "api_key": "do-not-capture"},
    )
    capture.close()
    run = normalize(path, title="Tool definitions")
    assert run.steps[0].tool_definitions == (definitions if capture_content else None)
    output = tmp_path / "report.html"
    render(run, prices, output)
    html = output.read_text()
    assert '<details class="tool-definitions">' in html
    assert "<script>alert(1)</script>" not in html
    assert "do-not-capture" not in path.read_text()
    if capture_content:
        assert "unused_tool" in html and "Search phrase" in html
        assert "&lt;script&gt;" in html
    else:
        assert "Tool definitions were not captured" in html
        assert "Private description" not in path.read_text() + html


def test_empty_and_legacy_tool_definitions(tmp_path, prices):
    """Explicitly empty bindings differ from absent definitions in older runs."""
    step = model_step()
    assert step.tool_definitions is None
    step.tool_definitions = []
    run = Run(id="r", title="No tools", status="ok", steps=[step])
    output = tmp_path / "report.html"
    render(run, prices, output)
    assert "No tools were bound to this request." in output.read_text()
