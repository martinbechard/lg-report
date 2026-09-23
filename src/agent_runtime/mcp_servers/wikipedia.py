"""Serve the existing Wikipedia retrieval tool to MCP clients over local stdio.

Open an already completed index once at startup. Ingestion stays separate, and
the model can supply only a search query, never a database path. Stdout belongs
to MCP; FastMCP writes its diagnostics to stderr. No chat model is called here.

Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

# AI attribution: Comments refined with AI assistance (Northstar).

import argparse
from pathlib import Path

from fastmcp import FastMCP

from agent_runtime.agents.wikipedia_rag_agent import WIKIPEDIA_INDEX_DIRECTORY
from agent_runtime.harness.rag_index import open_index
from agent_runtime.tools.semantic_search_wikipedia import build_search_tool


def build_server(collection) -> FastMCP:
    """Make local Wikipedia evidence available to clients through MCP tool calls.

    ``collection`` must already be opened and validated by the caller. The
    returned server owns tool registration but does not close the collection or
    start transport; :func:`main` performs the stdio lifecycle.
    """
    server = FastMCP("Wikipedia RAG")
    # Constructing the tool captures the collection but performs no search.
    # Its name/description are discovery metadata; only a later MCP tools/call
    # executes search.func and returns passage JSON to the requesting client.
    search = build_search_tool(collection)
    # FastMCP registers the underlying Python callable, reusing the LangChain
    # tool's public description. Registration itself starts no transport.
    server.tool(
        name=search.name,
        description=search.description,
        annotations={"readOnlyHint": True, "destructiveHint": False},
    )(search.func)
    return server


def main() -> None:
    """Run the Wikipedia service so MCP clients can request indexed evidence.

    Arguments are trusted process configuration. ``open_index`` raises for a
    missing or incomplete index, keeping startup failure visible. FastMCP owns
    the blocking stdio loop and shutdown after this function hands it the server.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--directory",
        type=Path,
        default=WIKIPEDIA_INDEX_DIRECTORY,
        help="Completed WikiText index directory; defaults to the RAG sample index.",
    )
    args = parser.parse_args()
    server = build_server(open_index(args.directory))
    # Control remains in FastMCP's request loop until shutdown. Each search is
    # handled against the opened collection; no chat agent runs in this process.
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
