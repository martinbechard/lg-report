"""Serve the existing Wikipedia retrieval tool to MCP clients over local stdio.

Open an already completed index once at startup. Ingestion stays separate, and
the model can supply only a search query, never a database path. Stdout belongs
to MCP; FastMCP writes its diagnostics to stderr. No chat model is called here.

Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

import argparse
from pathlib import Path

from fastmcp import FastMCP

from lg_report.agents.wikipedia_rag_agent import WIKIPEDIA_INDEX_DIRECTORY
from lg_report.platform.rag_index import open_index
from lg_report.tools.search_wikipedia import build_search_tool


def build_server(collection) -> FastMCP:
    """Publish the same bounded retrieval function used by the local RAG agent."""
    server = FastMCP("Wikipedia RAG")
    search = build_search_tool(collection)
    server.tool(
        name=search.name,
        description=search.description,
        annotations={"readOnlyHint": True, "destructiveHint": False},
    )(search.func)
    return server


def main() -> None:
    """Let the operator select an index, validate it, then serve until shutdown."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--directory",
        type=Path,
        default=WIKIPEDIA_INDEX_DIRECTORY,
        help="Completed WikiText index directory; defaults to the RAG sample index.",
    )
    args = parser.parse_args()
    server = build_server(open_index(args.directory))
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
