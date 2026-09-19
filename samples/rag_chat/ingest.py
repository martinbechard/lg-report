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
    """Build the persistent index as an explicit, inspectable preparation step.

    The directory and passage limit are command-line inputs so a maintainer can
    choose storage and cost deliberately. ``build_index`` owns downloading,
    embedding, and manifest generation; this script only parses options and
    prints its returned statistics. It must finish before the chat sample runs.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=WIKIPEDIA_INDEX_DIRECTORY)
    parser.add_argument("--max-passages", type=int, default=DEFAULT_PASSAGES)
    args = parser.parse_args()
    # This call performs preparation now, including any required download and
    # embedding. Its return is index statistics, not retrieved agent evidence;
    # failures propagate so a partial build cannot print a success summary.
    print(json.dumps(build_index(args.directory, args.max_passages), indent=2))


if __name__ == "__main__":
    main()
