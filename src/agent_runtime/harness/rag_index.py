"""Build and reopen a persistent Chroma index of a reproducible Wikipedia subset.

Download and embed locally, outside the chat model and reporting token budget.
The dataset revision, tokenizer, chunk settings, and prefix limit are recorded so
re-running resumes the same index rather than mixing incompatible data. A final
manifest marks a complete index; chat refuses to query a partially built one.
Design and commands: samples/rag_chat/README.md.
AI attribution: Generated with AI assistance.

Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

import json
import re
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from huggingface_hub import hf_hub_download
from pyarrow import parquet
from tokenizers import Tokenizer

from agent_runtime.harness.rag_archives import restore_archive, restore_default_index

DATASET = "Salesforce/wikitext"
REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
TOKENIZER_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
COLLECTION = "wikitext103"
CHUNK_TOKENS = 220
OVERLAP_TOKENS = 32
DEFAULT_PASSAGES = 20_000


def file_bytes(directory: Path) -> int:
    """Measure the local index footprint for ingestion evidence.

    Return logical regular-file bytes under ``directory``, not allocated disk
    blocks or an estimate of chat-model token usage.

    Symlinks are excluded because bundled/cache layouts may point at shared data;
    counting their targets would overstate the artifact size recorded in reports.
    """
    return sum(
        p.stat().st_size
        for p in directory.rglob("*")
        if p.is_file() and not p.is_symlink()
    )


def _write_json(path: Path, value: dict) -> None:
    """Keep readers from seeing half-written index metadata.

    Serialize ``value`` to a sibling temporary file and replace ``path``; return
    no value. This publishes one file, not a transaction across index contents.

    The temporary sibling is replaced only after serialization succeeds, so an
    interruption cannot leave a truncated config or manifest that looks valid.
    """
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def articles(paths):
    """Recover article boundaries so retrieved passages retain source attribution.

    ``paths`` is the ordered iterable of local Parquet shards. Yield
    ``(title, source_row, text)`` tuples lazily as the caller advances iteration;
    rows are counted across shards, including blank lines and headings.

    WikiText stores lines, not documents. Only single-level headings begin a new
    article; section headings remain part of that article. Carry state across
    shard boundaries because a file boundary is not necessarily an article end.
    """
    title, start_row, lines = "Untitled preamble", 0, []
    row_number = 0
    for path in paths:
        for batch in parquet.ParquetFile(path).iter_batches(
            batch_size=1024, columns=["text"]
        ):
            for text in batch.column(0).to_pylist():
                heading = re.fullmatch(r"= ([^=].*?) =", text.strip())
                if heading:
                    # Emit the old article before changing its attribution.
                    if lines:
                        yield title, start_row, "\n".join(lines)
                    title, start_row, lines = heading.group(1), row_number, []
                elif text.strip():
                    lines.append(text.strip())
                row_number += 1
    if lines:
        # No following heading exists to flush the final article.
        yield title, start_row, "\n".join(lines)


def chunks(text: str, tokenizer: Tokenizer):
    """Split in the embedding tokenizer's units to avoid silent model truncation.

    Overlap preserves evidence crossing a boundary. At most 220 content tokens
    leaves room for the MiniLM model's special tokens within its 256-token limit.
    ``text`` is one article; ``tokenizer`` must match the embedding model.
    Yield ``(start, end, decoded_text)`` with token offsets and an exclusive end.
    Decoded text is what Chroma actually embeds and later retrieves.
    """
    token_ids = tokenizer.encode(text, add_special_tokens=False).ids
    for start in range(0, len(token_ids), CHUNK_TOKENS - OVERLAP_TOKENS):
        end = min(start + CHUNK_TOKENS, len(token_ids))
        yield start, end, tokenizer.decode(token_ids[start:end])
        if end == len(token_ids):
            # Do not emit an extra fragment containing only an already-covered tail.
            break


def open_index(directory: Path):
    """Make a completed local evidence collection available for retrieval.

    Return a Chroma collection handle; opening it does not query documents.

    ``directory`` must contain a completed manifest whose passage count matches
    Chroma. The same local embedding function is used for documents and questions.
    No collection is created here: a miss must not look like successful retrieval
    from an empty database. Client lifecycle is process-owned for this local app.
    """
    restore_default_index(directory)
    manifest_path = directory / "manifest.json"
    if not manifest_path.exists():
        raise ValueError(
            "Build the index first: uv run python -m samples.rag_chat.ingest"
        )
    manifest = json.loads(manifest_path.read_text())
    client = chromadb.PersistentClient(path=str(directory / "chroma"))
    collection = client.get_collection(
        COLLECTION, embedding_function=DefaultEmbeddingFunction()
    )
    if collection.count() != manifest["passages"]:
        raise ValueError("Chroma count does not match completed ingestion manifest")
    return collection


def build_index(directory: Path, max_passages: int = DEFAULT_PASSAGES):
    """Prepare reusable Wikipedia evidence for local retrieval in chat workflows.

    Build or reuse a deterministic passage prefix and return its manifest of
    dataset settings, counts, and size evidence. This does not run a chat agent.

    ``directory`` is the durable cache. ``max_passages`` limits local embedding
    time, not the downloaded corpus. Positive
    values select that many passages; requests larger than the corpus stop at EOF.
    Downloads are cached. Completed indexes are reused; interrupted indexes resume
    from their durable sequential prefix. Use a new directory for other settings.
    No paid embedding or chat request is made. Transport/disk/model errors propagate.
    """
    if max_passages <= 0:
        raise ValueError("max_passages must be positive")
    if max_passages == DEFAULT_PASSAGES:
        restore_default_index(directory)
    directory.mkdir(parents=True, exist_ok=True)
    config = {
        "dataset": DATASET,
        "revision": REVISION,
        "split": "train",
        "embedding": "Chroma DefaultEmbeddingFunction/all-MiniLM-L6-v2",
        "tokenizer_revision": TOKENIZER_REVISION,
        "chunk_tokens": CHUNK_TOKENS,
        "overlap_tokens": OVERLAP_TOKENS,
        "max_passages": max_passages,
    }
    config_path = directory / "config.json"
    # Resuming requires the same source, chunking, and prefix limit. Otherwise
    # durable passage IDs could refer to different text under the same index.
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("Index settings changed; choose a new directory")
    _write_json(config_path, config)
    manifest_path = directory / "manifest.json"
    if manifest_path.exists():
        open_index(directory)
        return json.loads(manifest_path.read_text())
    if not (directory / "dataset").exists():
        restore_archive("dataset", directory)
    paths = []
    for shard in range(2):
        local_shard = (
            directory
            / "dataset"
            / "wikitext-103-raw-v1"
            / f"train-{shard:05d}-of-00002.parquet"
        )
        paths.append(
            str(local_shard)
            if local_shard.exists()
            else hf_hub_download(
                DATASET,
                repo_type="dataset",
                revision=REVISION,
                filename=f"wikitext-103-raw-v1/train-{shard:05d}-of-00002.parquet",
                local_dir=directory / "dataset",
            )
        )
    tokenizer_path = hf_hub_download(
        "sentence-transformers/all-MiniLM-L6-v2",
        revision=TOKENIZER_REVISION,
        filename="tokenizer.json",
        local_dir=directory / "tokenizer",
    )
    tokenizer = Tokenizer.from_file(tokenizer_path)
    # Chunk the full article ourselves. Tokenizer truncation or padding here
    # would lose source text or count artificial content before passage slicing.
    tokenizer.no_truncation()
    tokenizer.no_padding()
    collection = chromadb.PersistentClient(
        path=str(directory / "chroma")
    ).get_or_create_collection(
        COLLECTION,
        embedding_function=DefaultEmbeddingFunction(),
        configuration={"hnsw": {"space": "cosine"}},
    )
    # Resume assumes this application wrote a contiguous prefix of sequential
    # IDs. A count alone cannot establish that invariant for an externally edited
    # collection; this is a sample-owned cache, not a general database repair.
    completed = collection.count()
    pending_ids, pending_texts, pending_metadata = [], [], []
    total_tokens = unique_tokens = total_text_bytes = passage_count = 0
    article_titles = set()

    def flush():
        """Persist a bounded passage batch so ingestion can make durable progress.

        Captured pending lists contain parallel IDs, texts, and attribution.
        Chroma upsert embeds the texts locally and stores the resulting vectors;
        it is external I/O, not merely queueing a later operation. Successful
        writes clear all three lists. Errors propagate with the lists retained;
        an empty batch performs no work. Return no value.
        """
        if pending_ids:
            # IDs follow source order; successful batches form a resumable prefix.
            collection.upsert(
                ids=pending_ids, documents=pending_texts, metadatas=pending_metadata
            )
            pending_ids.clear()
            pending_texts.clear()
            pending_metadata.clear()

    for title, source_row, text in articles(paths):
        for start, end, passage in chunks(text, tokenizer):
            if passage_count >= max_passages:
                break
            identifier = f"passage-{passage_count:08d}"
            passage_count += 1
            total_tokens += end - start
            unique_tokens += end - start if start == 0 else end - start - OVERLAP_TOKENS
            total_text_bytes += len(passage.encode())
            article_titles.add((title, source_row))
            if passage_count <= completed:
                # Already persisted before interruption: recount, but do not re-embed.
                continue
            pending_ids.append(identifier)
            pending_texts.append(passage)
            pending_metadata.append(
                {
                    "title": title,
                    "source_row": source_row,
                    "token_start": start,
                    "token_end": end,
                    "source": f"https://huggingface.co/datasets/{DATASET}/tree/{REVISION}/wikitext-103-raw-v1",
                    "license": "CC-BY-SA-3.0",
                }
            )
            if len(pending_ids) == 128:
                flush()
                if passage_count % 1024 == 0:
                    print(
                        f"Indexed {passage_count:,}/{max_passages:,} passages",
                        flush=True,
                    )
        if passage_count >= max_passages:
            break
    flush()
    manifest = {
        **config,
        "passages": passage_count,
        "articles": len(article_titles),
        "embedding_tokens_with_overlap": total_tokens,
        "embedding_tokens_without_overlap": unique_tokens,
        "indexed_text_bytes_with_overlap": total_text_bytes,
        "downloaded_parquet_bytes": sum(Path(p).stat().st_size for p in paths),
        "chroma_bytes_at_completion": file_bytes(directory / "chroma"),
    }
    if collection.count() != passage_count:
        raise ValueError("Index count mismatch; refusing to publish completion")
    # Publishing completion is the final step after all batches and the count
    # check. A failed build leaves no new completion manifest for chat to trust.
    _write_json(manifest_path, manifest)
    return manifest
