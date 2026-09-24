<!-- Copyright (c) 2026 Martin.Bechard@DevConsult.ca -->
# Optional Wikipedia RAG download

The `index` archive holds the completed 20,000-passage Chroma index, its build
configuration, and completion manifest. The `dataset` archive holds both original
WikiText-103 raw training Parquet shards. XZ archives are split into at most
48 MiB pieces distributed as GitHub Release attachments. The payloads are not
tracked in Git; the checksum catalog and attribution remain in the repository. `archives.json` records
ordered pieces, byte sizes, and SHA-256 checksums. Keep every piece together.

From the repository root, download and verify every piece using Python 3.11+:

```sh
python scripts/download_rag.py
```

The downloader uses the latest release and skips existing verified pieces.
For an older checkout, use `--tag v0.1.0` (or its matching release).
Interrupted downloads can be retried. The prebuilt UI needs no separate download.

The initial snapshot compresses the approximately 225 MiB Chroma database to
95.1 MiB (two pieces), and the 299.5 MiB dataset to 246.7 MiB (six pieces).

The first RAG chat or default ingestion run restores the index into the ignored
`.cache/lg-report/rag/` directory. Subsequent runs reuse it. Existing partial
indexes are never overwritten. To recover an interrupted extraction, move that
partial cache aside and retry. A custom ingestion directory extracts the dataset
and builds its own index. Missing pieces identify the download command; corrupt archive pieces fail visibly.
If no catalog is installed, the ingestion code can use its upstream download path.

The tokenizer and embedding model are not bundled. Query embeddings may download
Chroma's MiniLM model on first use; rebuilding also obtains the pinned tokenizer.
The sample application is used from a source checkout; wheels do not bundle these
large repository assets.

To update the snapshots, finish ingestion and **stop every process using Chroma**,
then run from the repository root:

```sh
uv run python scripts/package_rag.py
```

Review and commit `archives.json` and license sidecars, then publish the referenced
pieces using the release instructions in [PIP-INSTALL.md](../../PIP-INSTALL.md).
Remove any old, unreferenced local pieces after repacking. Do not copy a database during writes.

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
