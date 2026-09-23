"""Retrieve source-labelled facts from small, separate teaching reference sets.

Each expert gets only its domain's tool. Retrieval uses case-insensitive phrase
matching, not embeddings or internet search. The references are deliberately
small: an unmatched topic returns an explicit miss instead of unrelated evidence.
Adding production RAG would replace retrieval here, not the agent/client loop.
AI attribution: Generated with AI assistance.

Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

from langchain_core.tools import tool

# These authored summaries are reference data, independent of sample questions
# and scripted answers. Source URLs let readers inspect the underlying evidence;
# invoking a tool reads this snapshot and does not fetch those URLs.
_REFERENCES = {
    "movies": [
        {
            "title": "Spirited Away",
            "aliases": ("spirited away", "miyazaki"),
            "text": "Spirited Away was directed by Hayao Miyazaki and released in 2001.",
            "source": "https://www.ghibli.jp/works/",
        },
    ],
    "sports": [
        {
            "title": "Standard basketball teams",
            "aliases": ("basketball", "five players"),
            "text": "Standard basketball has two teams of five players on court. This is not the separate 3x3 format.",
            "source": "https://about.fiba.basketball/en/our-sport/basketball",
        },
    ],
    "history": [
        {
            "title": "Opening of the Berlin Wall",
            "aliases": ("berlin wall", "mauerfall"),
            "text": "The Berlin Wall opened on 9 November 1989. The opening and subsequent physical demolition are distinct events.",
            "source": "https://www.stiftung-berliner-mauer.de/de/ueber-uns/leichte-sprache/geschichte",
        },
    ],
}

# The mapping is a private, read-only-by-convention corpus boundary. Wrappers
# choose the domain, so a model argument cannot broaden one expert's capability.


def _retrieve(domain: str, query: str) -> str:
    """Supply an expert with relevant local evidence and traceable source labels.

    domain is fixed by each tool wrapper; the model cannot select another corpus
    through an argument. Matching aliases within a query makes the small example
    usable with either a topic phrase or a full question, without a search service.
    The helper assumes a known key from a wrapper; an invalid internal key raises
    normally so programming errors are not disguised as retrieval misses.
    """
    normalized_query = query.casefold().strip()
    matches = []
    for reference in _REFERENCES[domain]:
        # A phrase match is required: returning the only document for every query
        # would hide retrieval failures and teach the model to cite irrelevant text.
        if any(alias in normalized_query for alias in reference["aliases"]):
            matches.append(
                f"{reference['title']}\n{reference['text']}\nSource: {reference['source']}"
            )
    if not matches:
        # No evidence is different from an answer of 'no'. Let the expert explain
        # the coverage limit or ask for clarification instead of inventing support.
        return f"No matching {domain} reference. This local teaching collection has limited coverage."
    return "\n\n".join(matches)


@tool(parse_docstring=True)
def search_movie_reference(query: str) -> str:
    """Look up local movie evidence by phrase; return sources or an explicit miss.

    Args:
        query: A movie topic or question containing a title or name to match.
            The small teaching collection covers Spirited Away and Miyazaki.
    """
    # Supply this expert's evidence before it composes an answer. The wrapper
    # fixes the corpus; query comes from the model's proposed search arguments.
    # Return passage text for the graph to carry as the tool observation.
    return _retrieve("movies", query)


@tool(parse_docstring=True)
def search_sports_reference(query: str) -> str:
    """Look up local sports evidence by phrase; return sources or an explicit miss.

    Args:
        query: A sports topic or question containing a phrase to match.
            The small teaching collection covers standard basketball team size.
    """
    # Supply this expert's evidence before it composes an answer. The wrapper
    # fixes the corpus; query comes from the model's proposed search arguments.
    # Return passage text for the graph to carry as the tool observation.
    return _retrieve("sports", query)


@tool(parse_docstring=True)
def search_history_reference(query: str) -> str:
    """Look up local history evidence by phrase; return sources or an explicit miss.

    Args:
        query: A historical topic or question containing a phrase to match.
            The small teaching collection covers the Berlin Wall opening.
    """
    # Supply this expert's evidence before it composes an answer. The wrapper
    # fixes the corpus; query comes from the model's proposed search arguments.
    # Return passage text for the graph to carry as the tool observation.
    return _retrieve("history", query)
