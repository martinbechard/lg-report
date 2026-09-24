"""Reproduce repeated forbidden writes without paying for a model loop.

Only model decisions are scripted. Real file tools reject the first three
requests and real middleware blocks the fourth. No scripted final response
stands in for the breaker: its termination message comes from LangChain.

AI attribution: Generated with AI assistance by Northstar.
Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

from langchain_core.messages import AIMessage

from agent_runtime.harness.simulated_model import MeteredDemoModel

# Author each user turn beside the model replies it should provoke. One user
# message can cause several model/tool exchanges before the agent stops.
SCENARIO = [
    {
        "user": (
            "Reproduce the defective plan: write /slugify.py containing '# Slug implementation pending'. "
            "Repeat the identical write after rejection so the circuit breaker can stop the loop."
        ),
        "responses": [
            # LangChain's AIMessage describes one model reply. Empty content
            # means no prose answer; tool_calls proposes an action for the agent
            # to execute later. Constructing this message does not write a file.
            AIMessage(content="", tool_calls=[{
                "name": "write_file",
                "args": {"file_path": "/slugify.py", "content": "# Slug implementation pending"},
                # A unique ID connects each proposal to its own tool result,
                # even though the requested path and content are identical.
                "id": f"forbidden-write-{attempt}",
            }])
            # Three rejected writes, then an attempt blocked by the call limit.
            for attempt in range(1, 5)
        ],
    }
]

# The client sends user messages; the simulated model supplies model replies.
# Both collections come from the same authored scenario, in conversation order.
USER_PROMPTS = [turn["user"] for turn in SCENARIO]
RESPONSES = [response for turn in SCENARIO for response in turn["responses"]]


def build_models(options):
    """Supply mock LLM objects through the sample catalog's factory callback.

    sample.json names this scripted_run module. In simulated mode the catalog
    imports it and explicitly calls script.build_models(options); the name is a
    required callback convention, not something Python discovers automatically.
    When the workflow asks for build_model(caller="worker"), the catalog returns
    the "worker" entry from this callback's dictionary. Live mode uses the
    configured LLM provider instead.

    ``options`` is a dictionary of workflow settings: the catalog merges defaults
    from sample.json with any options supplied for the current run, with run
    options taking precedence. The shared scripted-model factory interface accepts
    this dictionary so samples can tailor their mock models to those settings.
    This sample has a fixed sequence of forbidden writes, so it ignores options.

    We script the bad decisions because a live model might correct its filename
    after a rejection, making the circuit-breaker lesson unreliable. Each call
    creates a fresh MeteredDemoModel: it supplies the prepared replies in order
    and adds simulated token usage for reports, without contacting a provider.

    Construction does not write a file or run the agent. During the run, the real
    file backend rejects the first three requests because /slugify.py is not an
    allowed path (/slug.py is). The workflow permits three write_file calls;
    request four demonstrates that its real limit middleware stops the loop
    before a fourth write reaches the backend. The stop is not a scripted reply.
    """
    # "worker" must match the workflow's build_model(caller="worker") lookup.
    # Each run gets its own model cursor and message copies, so runtime changes
    # cannot modify the authored scenario or another run's prepared responses.
    return {"worker": MeteredDemoModel(responses=[
        response.model_copy(deep=True) for response in RESPONSES
    ])}
