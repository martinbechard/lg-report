"""Define the movies expert through its instructions and reference tool.

The expert can use the same model as the other roles. Its specialization comes
from the domain instructions and access to a separate local evidence collection.
It cannot delegate or search the other experts' collections.
See docs/chat-composition.md for workflow composition.
AI attribution: Generated with AI assistance.

Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langgraph.graph.state import CompiledStateGraph

from lg_report.tools.domain_reference import search_movie_reference

NAME = "movie_expert"
# Registration metadata is the dispatcher's capability boundary; it does not
# grant this expert access to another domain's reference or conversation state.
DESCRIPTION = (
    "Specialist for questions about movies, with its own local reference lookup."
)
SYSTEM_PROMPT = (
    "You are the movies expert. Use search_movie_reference to retrieve evidence "
    "before answering the delegated question. Base factual claims on the returned "
    "passages and cite their source URLs. If no relevant evidence is found, explain "
    "the collection's limits instead of inventing an answer or claiming web research. "
    "Answer concisely. The collection is a small local teaching reference, not a "
    "complete or continually updated source."
)


def build_agent(model: BaseChatModel) -> CompiledStateGraph:
    """Prepare a movie specialist to answer delegated questions with local evidence.

    model is a provider adapter or scripted test model, not domain documents.
    Construction invokes neither model nor tool; the graph runs on an assignment
    from the dispatcher. Its tool result becomes input to the next model request.
    """
    # During invocation the graph executes proposed lookup calls, adds their
    # strings as ToolMessages, and calls the model again to compose an answer.
    # ToolMessage is LangChain's observation container; the graph's own return
    # is a state dictionary with message history, including the final AIMessage.
    # Only the domain-specific lookup is registered. Using the same underlying
    # model across experts does not grant access to another expert's tools/context.
    return create_agent(
        model=model,
        tools=[search_movie_reference],
        name=NAME,
        system_prompt=SYSTEM_PROMPT,
    )
