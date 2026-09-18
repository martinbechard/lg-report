"""Download the corpus and build its persistent vector index before running chat.

Ingestion is explicit so starting a conversation never unexpectedly downloads
hundreds of megabytes or embeds thousands of passages. See this sample's README.
AI attribution: Generated with AI assistance.

Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

import argparse
import json
from pathlib import Path

from lg_report.agents.wikipedia_rag_agent import WIKIPEDIA_INDEX_DIRECTORY
from lg_report.platform.rag_index import DEFAULT_PASSAGES, build_index


def main():
    """Expose dataset location and size; print only progress and final statistics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=WIKIPEDIA_INDEX_DIRECTORY)
    parser.add_argument("--max-passages", type=int, default=DEFAULT_PASSAGES)
    args = parser.parse_args()
    print(json.dumps(build_index(args.directory, args.max_passages), indent=2))


if __name__ == "__main__":
    main()
