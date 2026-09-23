"""Expose bounded semantic retrieval as an agent tool, not a full-corpus prompt.

The agent or MCP server injects a completed Chroma collection. Each call returns at most
four passages with durable IDs and source metadata for the answer's citations.
Local embedding and search make no provider API calls; retrieved text becomes
billable input only when the chat model consumes it in its next request.
AI attribution: Generated with AI assistance.

Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

import json

from langchain_core.tools import tool


def build_search_tool(collection):
    """Give an agent bounded passage retrieval from the application's chosen corpus.

    ``collection`` is an already-open local Chroma-compatible collection owned
    by the caller. Binding captures that handle for the tool lifetime; this
    function does not close it, ingest data, or catch storage errors.
    """

    # The decorator uses the docstring below as a model-facing description.
    # Keep implementation explanations in comments to preserve that contract.
    # Call this tool when factual context is needed; its outcome is a small
    # candidate evidence set with IDs the agent can use for citations.
    @tool(parse_docstring=True)
    def semantic_search_wikipedia(query: str) -> str:
        """Search indexed WikiText and return up to four source-labelled passages.

        The corpus is a historical Wikipedia subset, not current news. Retrieved
        passages are candidates; check relevance before using them as evidence.

        Args:
            query: A nonempty, focused search question or phrase of at most
                500 characters describing the information to retrieve.
        """
        # Validate before touching the collection so malformed model input is a
        # cheap, deterministic tool result and cannot trigger an empty query.
        if not query.strip():
            return json.dumps(
                {"passages": [], "note": "Supply a nonempty search question."}
            )
        if len(query) > 500:
            # Keep query embedding bounded; do not silently truncate a pasted file.
            return json.dumps(
                {
                    "passages": [],
                    "note": "Use a focused query of at most 500 characters.",
                }
            )
        # This executes retrieval now: Chroma embeds the one supplied query and
        # ranks local passages. It is a database result, not the response from an
        # agent invocation and not a list of the model's proposed tool calls.
        results = collection.query(
            query_texts=[query],
            n_results=min(4, collection.count()),
            include=["documents", "metadatas", "distances"],
        )
        # Chroma returns parallel arrays. strict zip detects a corrupt or custom
        # collection response instead of silently dropping a passage or metadata.
        # Each outer list represents one query. We supplied exactly one, so [0]
        # selects that query's hits; matching positions in the inner arrays refer
        # to one passage. Distance is a similarity measure, not factual certainty.
        passages = []
        for identifier, text, metadata, distance in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
            strict=True,
        ):
            passages.append(
                {
                    "id": identifier,
                    "text": text,
                    **metadata,
                    "cosine_distance": distance,
                }
            )
        # Serialize the actual evidence for the tool transport. The consuming
        # agent receives this JSON string as an observation and decides which
        # passages support its answer; retrieval itself does not write an answer.
        return json.dumps(
            {
                "passages": passages,
                "note": "Nearest passages are candidates, not proof of relevance; abstain if they do not answer the question.",
            }
        )

    return semantic_search_wikipedia
