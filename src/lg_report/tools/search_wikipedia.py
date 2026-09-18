"""Expose bounded semantic retrieval as an agent tool, not a full-corpus prompt.

The workflow injects a completed Chroma collection. Each call returns at most
four passages with durable IDs and source metadata for the answer's citations.
Local embedding and search make no provider API calls; retrieved text becomes
billable input only when the chat model consumes it in its next request.
AI attribution: Generated with AI assistance.

Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

import json

from langchain_core.tools import tool


def build_search_tool(collection):
    """Bind one corpus to a tool; the model cannot choose paths or other databases."""

    @tool(parse_docstring=True)
    def search_wikipedia(query: str) -> str:
        """Search indexed WikiText and return up to four source-labelled passages.

        The corpus is a historical Wikipedia subset, not current news. Retrieved
        passages are candidates; check relevance before using them as evidence.

        Args:
            query: A nonempty, focused search question or phrase of at most
                500 characters describing the information to retrieve.
        """
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
        results = collection.query(
            query_texts=[query],
            n_results=min(4, collection.count()),
            include=["documents", "metadatas", "distances"],
        )
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
        return json.dumps(
            {
                "passages": passages,
                "note": "Nearest passages are candidates, not proof of relevance; abstain if they do not answer the question.",
            }
        )

    return search_wikipedia
