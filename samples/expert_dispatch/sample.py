"""Author the expert dispatch example in chronological conversation order.

SAMPLE declares discovery metadata alongside this scenario. Model factories
create fresh simulated models only when called; live mode uses the provider.

User messages and AI replies share one script, including multiple model/tool
steps within a user turn. Tool entries document expected observations only;
the workflow executes the actual tools. Model choices remain simulated.
AI attribution: Generated with AI assistance by Northstar.
Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

from agent_runtime.harness.model_factory import client_prompts, model_responses
from agent_runtime.harness.shared_simulated_model import SharedSimulatedModel

# Discovery reads this metadata without constructing a model.
SAMPLE = {
    "id": "expert_dispatch",
    "name": "Expert dispatch",
    "description": "Movie, sports and history specialists.",
}

CONVERSATION = [
    {"role": "client", "content": "Who directed Spirited Away?"},
    {
        "role": "ai",
        "content": "",
        "tool_calls": [
            {
                "name": "task",
                "args": {
                    "subagent_type": "movie_expert",
                    "description": "Who directed Spirited Away?",
                },
                "id": "dispatch-1",
            }
        ],
    },
    {
        "role": "movie_expert",
        "content": "",
        "tool_calls": [
            {
                "name": "search_movie_reference",
                "args": {"query": "Who directed Spirited Away?"},
                "id": "retrieve-1",
            }
        ],
    },
    {
        "role": "tool",
        "name": "search_movie_reference",
        "tool_call_id": "retrieve-1",
        "content": "Expected: actual search_movie_reference result for {'query': 'Who directed Spirited Away?'}; "
        "supplied by the running tool.",
    },
    {
        "role": "movie_expert",
        "content": "Hayao Miyazaki directed Spirited Away, released in 2001. Source: https://www.ghibli.jp/works/",
    },
    {
        "role": "tool",
        "name": "task",
        "tool_call_id": "dispatch-1",
        "content": "Hayao Miyazaki directed Spirited Away, released in 2001. Source: https://www.ghibli.jp/works/",
    },
    {
        "role": "ai",
        "content": "Hayao Miyazaki directed Spirited Away, released in 2001. Source: https://www.ghibli.jp/works/",
    },
    {
        "role": "client",
        "content": "How many players does a basketball team have on court?",
    },
    {
        "role": "ai",
        "content": "",
        "tool_calls": [
            {
                "name": "task",
                "args": {
                    "subagent_type": "sports_expert",
                    "description": "How many players does a basketball team have on court?",
                },
                "id": "dispatch-2",
            }
        ],
    },
    {
        "role": "sports_expert",
        "content": "",
        "tool_calls": [
            {
                "name": "search_sports_reference",
                "args": {
                    "query": "How many players does a basketball team have on court?"
                },
                "id": "retrieve-2",
            }
        ],
    },
    {
        "role": "tool",
        "name": "search_sports_reference",
        "tool_call_id": "retrieve-2",
        "content": "Expected: actual search_sports_reference result for {'query': 'How many players does a "
        "basketball team have on court?'}; supplied by the running tool.",
    },
    {
        "role": "sports_expert",
        "content": "A basketball team has five players on court during normal play. Source: "
        "https://about.fiba.basketball/en/our-sport/basketball",
    },
    {
        "role": "tool",
        "name": "task",
        "tool_call_id": "dispatch-2",
        "content": "A basketball team has five players on court during normal play. Source: "
        "https://about.fiba.basketball/en/our-sport/basketball",
    },
    {
        "role": "ai",
        "content": "A basketball team has five players on court during normal play. Source: "
        "https://about.fiba.basketball/en/our-sport/basketball",
    },
    {"role": "client", "content": "In which year did the Berlin Wall fall?"},
    {
        "role": "ai",
        "content": "",
        "tool_calls": [
            {
                "name": "task",
                "args": {
                    "subagent_type": "history_expert",
                    "description": "In which year did the Berlin Wall fall?",
                },
                "id": "dispatch-3",
            }
        ],
    },
    {
        "role": "history_expert",
        "content": "",
        "tool_calls": [
            {
                "name": "search_history_reference",
                "args": {"query": "In which year did the Berlin Wall fall?"},
                "id": "retrieve-3",
            }
        ],
    },
    {
        "role": "tool",
        "name": "search_history_reference",
        "tool_call_id": "retrieve-3",
        "content": "Expected: actual search_history_reference result for {'query': 'In which year did the Berlin "
        "Wall fall?'}; supplied by the running tool.",
    },
    {
        "role": "history_expert",
        "content": "The Berlin Wall opened on 9 November 1989, the opening preceded its physical demolition. "
        "Source: https://www.stiftung-berliner-mauer.de/de/ueber-uns/leichte-sprache/geschichte",
    },
    {
        "role": "tool",
        "name": "task",
        "tool_call_id": "dispatch-3",
        "content": "The Berlin Wall opened on 9 November 1989, the opening preceded its physical demolition. "
        "Source: https://www.stiftung-berliner-mauer.de/de/ueber-uns/leichte-sprache/geschichte",
    },
    {
        "role": "ai",
        "content": "The Berlin Wall opened on 9 November 1989, the opening preceded its physical demolition. "
        "Source: https://www.stiftung-berliner-mauer.de/de/ueber-uns/leichte-sprache/geschichte",
    },
]
USER_PROMPTS = client_prompts(CONVERSATION)
# Keep the compact case view used by integration tests derived from the script.
_dispatcher = model_responses(CONVERSATION)
CASES = [
    (
        request.tool_calls[0]["args"]["subagent_type"],
        request.tool_calls[0]["args"]["description"],
        answer.content,
    )
    for request, answer in zip(_dispatcher[::2], _dispatcher[1::2], strict=True)
]
del _dispatcher


def make_simulated_model():
    """Route the shared model's bound tools to role replies from one transcript.

    Each role retains an independent response cursor and usage ledger. Actual
    delegation and retrieval still execute in the graph between model calls.
    """
    return SharedSimulatedModel(
        scripts={
            "task": model_responses(CONVERSATION),
            "search_movie_reference": model_responses(CONVERSATION, "movie_expert"),
            "search_sports_reference": model_responses(CONVERSATION, "sports_expert"),
            "search_history_reference": model_responses(CONVERSATION, "history_expert"),
        },
        metadata={
            "report_description": "Execute this agent's role using the shared LLM and its bound tool.",
            "report_effort": "light",
        },
    )


def build_scripted_models(options):
    """Retain shared-model routing; this fixed script ignores workflow options."""
    return {"workflow": make_simulated_model()}
