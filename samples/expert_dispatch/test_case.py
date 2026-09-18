"""Author a reproducible three-domain conversation without provider calls.

These fixed model decisions test delegation plumbing, not natural-language
classification ability. Live mode uses the same workflow with provider models.
Each expert scripts a retrieval request; the actual tool supplies the evidence.
No agent or workflow imports this module or knows the questions in advance.
AI attribution: Generated with AI assistance.

Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

from langchain_core.messages import AIMessage

from lg_report.platform.shared_simulated_model import SharedSimulatedModel

CASES = [
    (
        "movie_expert",
        "Who directed Spirited Away?",
        "Hayao Miyazaki directed Spirited Away, released in 2001. Source: https://www.ghibli.jp/works/",
    ),
    (
        "sports_expert",
        "How many players does a basketball team have on court?",
        "A basketball team has five players on court during normal play. Source: https://about.fiba.basketball/en/our-sport/basketball",
    ),
    (
        "history_expert",
        "In which year did the Berlin Wall fall?",
        "The Berlin Wall opened on 9 November 1989, the opening preceded its physical demolition. Source: https://www.stiftung-berliner-mauer.de/de/ueber-uns/leichte-sprache/geschichte",
    ),
]
USER_PROMPTS = [question for _, question, _ in CASES]


def make_simulated_model():
    """Return one offline LLM with role-specific scripts keyed by available tool.

    The shared model receives each graph's own messages and tools. Script state
    stays separate so a specialist neither consumes dispatcher answers nor gains
    its cached history. Actual task and retrieval tools still run in LangGraph.
    """
    dispatcher_responses = []
    scripts = {}
    retrieval_tools = {
        "movie_expert": "search_movie_reference",
        "sports_expert": "search_sports_reference",
        "history_expert": "search_history_reference",
    }
    for number, (expert, question, answer) in enumerate(CASES, 1):
        dispatcher_responses.extend(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "task",
                            "args": {"subagent_type": expert, "description": question},
                            "id": f"dispatch-{number}",
                        }
                    ],
                ),
                AIMessage(content=answer),
            ]
        )
        tool_name = retrieval_tools[expert]
        scripts[tool_name] = [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": tool_name,
                        "args": {"query": question},
                        "id": f"retrieve-{number}",
                    }
                ],
            ),
            AIMessage(content=answer),
        ]
    scripts["task"] = dispatcher_responses
    return SharedSimulatedModel(
        scripts=scripts,
        metadata={
            "report_description": "Execute this agent's role using the shared LLM and its bound tool.",
            "report_effort": "light",
        },
    )
