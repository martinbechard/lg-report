<!-- Copyright (c) 2026 Martin.Bechard@DevConsult.ca -->
# Bundled Wikipedia RAG assets

The `index` archive holds the completed 20,000-passage Chroma index, its build
configuration, and completion manifest. The `dataset` archive holds both original
WikiText-103 raw training Parquet shards. XZ archives are split into at most
48 MiB pieces so individual Git objects stay manageable. `archives.json` records
ordered pieces, byte sizes, and SHA-256 checksums. Keep every piece together.

The initial snapshot compresses the approximately 225 MiB Chroma database to
95.1 MiB (two pieces), and the 299.5 MiB dataset to 246.7 MiB (six pieces).

The first RAG chat or default ingestion run restores the index into the ignored
`.cache/lg-report/rag/` directory. Subsequent runs reuse it. Existing partial
indexes are never overwritten. To recover an interrupted extraction, move that
partial cache aside and retry. A custom ingestion directory extracts the dataset
and builds its own index. Missing bundles fall back to explicit ingestion and
downloads; corrupt or missing archive pieces fail visibly.

The tokenizer and embedding model are not bundled. Query embeddings may download
Chroma's MiniLM model on first use; rebuilding also obtains the pinned tokenizer.
The sample application is used from a source checkout; wheels do not bundle these
large repository assets.

To update the snapshots, finish ingestion and **stop every process using Chroma**,
then run from the repository root:

```sh
uv run python scripts/package_rag.py
```

Review `archives.json` and commit the new referenced pieces together. Remove any
old, unreferenced pieces after repacking. Do not copy a database during writes.

## Source attribution and content license

Source: [Salesforce/WikiText](https://huggingface.co/datasets/Salesforce/wikitext),
revision `b08601e04326c79dfdd32d625aee71d232d685c3`, configuration
`wikitext-103-raw-v1`, both training shards. WikiText was assembled by Stephen
Merity, Caiming Xiong, James Bradbury, and Richard Socher from Wikipedia articles.
The original text is attributed to its Wikipedia contributors. The dataset's
[CC BY-SA 3.0 license](https://creativecommons.org/licenses/by-sa/3.0/) remains in
effect for the corpus and derived text; the software's MIT license does not
replace it. Chroma metadata retains article titles, source revision links, and
license identifiers for retrieved passages.

The Parquet shards are unchanged. The derived index reconstructs articles,
tokenizes and decodes overlapping 220-token chunks, and stores the first 20,000
passages with local MiniLM embeddings. Martin's copyright covers the packaging
and original project work, not the underlying Wikipedia text.
