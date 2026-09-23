"""Build independent reference experts from one reusable agent implementation.

Movies, sports, and history differ in role text and evidence tools, not in their
execution algorithm. Definitions stay at the agent level; the workflow selects
roles and supplies execution parameters. Each build creates a separate graph
with one domain tool and the original report/delegation identity.

AI attribution: Generated with AI assistance by Northstar.
Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

from dataclasses import dataclass

from langchain.agents import create_agent
from langchain_core.tools import BaseTool
from langgraph.graph.state import CompiledStateGraph

from agent_runtime.tools.search_reference import (
    search_history_reference,
    search_movie_reference,
    search_sports_reference,
)


@dataclass(frozen=True)
class ExpertDefinition:
    """Describe a role without constructing its model loop or loading evidence."""

    name: str  # Stable task routing and report identity.
    description: str  # Helps the dispatcher select this role.
    system_prompt: str  # Domain instructions belong to the agent.
    tool: BaseTool  # Only this domain's evidence collection is available.


MOVIE = ExpertDefinition(
    name="movie_expert",
    description="Specialist for questions about movies, with its own local reference lookup.",
    system_prompt="You are the movies expert. Use search_movie_reference to retrieve evidence before answering the delegated question. Base factual claims on the returned passages and cite their source URLs. If no relevant evidence is found, explain the collection's limits instead of inventing an answer or claiming web research. Answer concisely. The collection is a small local teaching reference, not a complete or continually updated source.",
    tool=search_movie_reference,
)

SPORTS = ExpertDefinition(
    name="sports_expert",
    description="Specialist for questions about sports, with its own local reference lookup.",
    system_prompt="You are the sports expert. Use search_sports_reference to retrieve evidence before answering the delegated question. Base factual claims on the returned passages and cite their source URLs. If no relevant evidence is found, explain the collection's limits instead of inventing an answer or claiming web research. Answer concisely. The collection is a small local teaching reference, not a complete or continually updated source.",
    tool=search_sports_reference,
)

HISTORY = ExpertDefinition(
    name="history_expert",
    description="Specialist for questions about history, with its own local reference lookup.",
    system_prompt="You are the history expert. Use search_history_reference to retrieve evidence before answering the delegated question. Base factual claims on the returned passages and cite their source URLs. If no relevant evidence is found, explain the collection's limits instead of inventing an answer or claiming web research. Answer concisely. The collection is a small local teaching reference, not a complete or continually updated source.",
    tool=search_history_reference,
)


def build_agent(parameters: dict, definition: ExpertDefinition) -> CompiledStateGraph:
    """Compile one isolated expert; model and lookup execute only on invocation."""
    return create_agent(
        **parameters,
        name=definition.name,
        system_prompt=definition.system_prompt,
        tools=[definition.tool],
    )
