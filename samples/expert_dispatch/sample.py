"""Author the expert dispatch example in chronological conversation order.

SAMPLE declares discovery metadata alongside this scenario. The shared catalog
constructs the offline model from CONVERSATION; live mode uses the provider.

User messages and AI replies share one script, including multiple model/tool
steps within a user turn. Tool entries document expected observations only;
the workflow executes the actual tools. Model choices remain simulated.
AI attribution: Generated with AI assistance by Northstar.
Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

# Discovery reads this metadata without constructing a model.
SAMPLE = {
    "id": "expert_dispatch",
    "name": "Expert dispatch",
    "description": "Movie, sports and history specialists.",
}

CONVERSATION = [
    {"role": "client", "content": "Who directed Spirited Away?"},
    {
        "role": "dispatcher_agent",
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
        "role": "dispatcher_agent",
        "content": "Hayao Miyazaki directed Spirited Away, released in 2001. Source: https://www.ghibli.jp/works/",
    },
    {
        "role": "client",
        "content": "How many players does a basketball team have on court?",
    },
    {
        "role": "dispatcher_agent",
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
        "role": "dispatcher_agent",
        "content": "A basketball team has five players on court during normal play. Source: "
        "https://about.fiba.basketball/en/our-sport/basketball",
    },
    {"role": "client", "content": "In which year did the Berlin Wall fall?"},
    {
        "role": "dispatcher_agent",
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
        "role": "dispatcher_agent",
        "content": "The Berlin Wall opened on 9 November 1989, the opening preceded its physical demolition. "
        "Source: https://www.stiftung-berliner-mauer.de/de/ueber-uns/leichte-sprache/geschichte",
    },
]
