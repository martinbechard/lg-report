"""Script two model/tool/model cycles to teach the complete tool-call lifecycle.

Each tool request has a distinct ID: LangGraph matches its result to that ID.
The graph executes the real local tool; this fixture scripts only the model.

AI attribution: Generated with AI assistance.

Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

# LangChain's AIMessage holds an assistant response; constructing it runs nothing.
# Its tool_calls, when present, are proposed names/arguments, not tool results.
from langchain_core.messages import AIMessage

from agent_runtime.harness.simulated_model import MeteredDemoModel


def make_simulated_model() -> MeteredDemoModel:
    """Demonstrate how tool observations enter the next model request across two turns.

    Return a fresh four-response fixture for two model/tool/model cycles.

    Reuse this model across both user turns so its response cursor and context
    ledger stay aligned. Only decisions and answers are scripted: the graph
    executes echo_tool and feeds its actual result to the next call.
    No provider requests occur while constructing or using this fixture.
    """
    # A tool-call response can have empty visible text. The structured call
    # still counts as model output and must be retained in conversation history.
    model = MeteredDemoModel(
        metadata={
            "report_effort": "fast",
            "report_description": "Request an echo, or compose the answer using the tool observation.",
        },
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "echo_tool",
                        "args": {"text": "ReAct"},
                        "id": "reference-1",
                    }
                ],
                response_metadata={
                    "model_name": "scripted-chat",
                    "finish_reason": "tool_calls",
                },
            ),
            AIMessage(
                content='The tool returned "Your input was: ReAct". This shows the model requesting a tool and then reading its result.',
                response_metadata={
                    "model_name": "scripted-chat",
                    "finish_reason": "stop",
                },
            ),
        ],
    )
    # This turn also demonstrates visible assistant text alongside a tool call.
    # Both belong to one model response; the tool result is a separate message.
    model.responses.extend(
        [
            AIMessage(
                content="I will echo the text through the local tool.",
                tool_calls=[
                    {
                        "name": "echo_tool",
                        "args": {"text": "Tool observations"},
                        "id": "reference-2",
                    }
                ],
                response_metadata={
                    "model_name": "scripted-chat",
                    "finish_reason": "tool_calls",
                },
            ),
            AIMessage(
                content='The tool returned "Your input was: Tool observations". The next model request includes that result. This echo repeats supplied text and adds no independent evidence.',
                response_metadata={
                    "model_name": "scripted-chat",
                    "finish_reason": "stop",
                },
            ),
        ]
    )
    return model


# User input belongs to the test scenario, never the agent definition.
USER_PROMPTS = [
    "Echo this text: ReAct",
    "Echo this text: Tool observations",
]

# The two prompts map to the two response pairs above. They are kept in the
# client fixture so the same workflow can accept console input independently.


def build_models(options):
    """Give the model factory fresh caller-specific scripted adapters for one run.

    The catalog supplies workflow options, not model instances. Actual tools
    still execute in the graph; these adapters author decisions and meter usage.
    """
    return {"workflow": make_simulated_model()}
