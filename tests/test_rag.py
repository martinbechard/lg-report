"""Check RAG boundaries without network downloads or paid model requests.

Small local Chroma indexes use a deterministic test embedding, not production
semantic vectors. The separate full-dataset smoke run verifies real MiniLM search.
AI attribution: Generated with AI assistance.

Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

import json

import chromadb
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from langchain_core.messages import AIMessage
from tokenizers import Tokenizer, models, pre_tokenizers

from lg_report.platform.conversation import Conversation, Request
from lg_report.platform.rag_index import articles, chunks, open_index
from lg_report.platform.simulated_model import MeteredDemoModel
from lg_report.platform.static_client import StaticClient
from lg_report.tools.search_wikipedia import build_search_tool
from lg_report.workflows.rag_chat import build_workflow


class TestEmbedding:
    """Separate two fixture topics so tests exercise real database query plumbing."""

    __test__ = False

    def __call__(self, input):
        return [
            [1.0, 0.0, 0.0] if "raven" in text else [0.0, 1.0, 0.0] for text in input
        ]

    def embed_query(self, input):
        return self(input)

    def name(self):
        return "test-topic-embedding"


def test_article_boundaries_across_files(tmp_path):
    first = tmp_path / "one.parquet"
    second = tmp_path / "two.parquet"
    pq.write_table(
        pa.table({"text": [" = First = \n", "One.", " = = Section = = ", "Two."]}),
        first,
    )
    pq.write_table(pa.table({"text": ["Three.", " = Second = ", "Four."]}), second)
    result = list(articles([first, second]))
    assert [row[0] for row in result] == ["First", "Second"]
    assert "Three." in result[0][2] and "Section" in result[0][2]
    assert result[1][1] == 5


def test_token_chunks_overlap_and_cover_tail():
    tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0, "word": 1}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    result = list(chunks(" ".join(["word"] * 441), tokenizer))
    assert [(start, end) for start, end, _ in result] == [
        (0, 220),
        (188, 408),
        (376, 441),
    ]
    assert all(len(tokenizer.encode(text).ids) <= 220 for _, _, text in result)
    assert list(chunks("", tokenizer)) == []


def test_chroma_persistence_tool_and_graph(tmp_path, monkeypatch):
    client = chromadb.PersistentClient(path=str(tmp_path / "db"))
    collection = client.create_collection(
        "test-rag", embedding_function=TestEmbedding()
    )
    collection.add(
        ids=["raven-1", "other-1"],
        documents=["A raven adapts to urban environments.", "A history passage."],
        metadatas=[
            {"title": "Raven", "source": "fixture"},
            {"title": "History", "source": "fixture"},
        ],
    )
    # Reopen by name: test persistence rather than an in-memory list masquerading as retrieval.
    reopened = client.get_collection("test-rag", embedding_function=TestEmbedding())
    tool = build_search_tool(reopened)
    result = json.loads(tool.invoke({"query": "raven"}))
    assert result["passages"][0]["id"] == "raven-1"
    assert len(result["passages"]) == 2
    assert json.loads(tool.invoke({"query": ""}))["passages"] == []
    assert json.loads(tool.invoke({"query": "x" * 501}))["passages"] == []
    model = MeteredDemoModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "search_wikipedia",
                        "args": {"query": "raven"},
                        "id": "lookup",
                    }
                ],
            ),
            AIMessage(content="Urban adaptation [raven-1]"),
        ]
    )
    from lg_report.agents import wikipedia_rag_agent

    opened_paths = []

    def open_fixture_index(directory):
        # Replace disk selection at the agent's boundary; the graph still uses
        # the real test collection and real retrieval tool internally.
        opened_paths.append(directory)
        return reopened

    monkeypatch.setattr(wikipedia_rag_agent, "open_index", open_fixture_index)
    final = Conversation(
        build_workflow(model), StaticClient([Request("Where does the raven live?")])
    ).invoke({}, {})
    assert opened_paths == [wikipedia_rag_agent.WIKIPEDIA_INDEX_DIRECTORY]
    observation = next(
        message for message in final["messages"] if message.type == "tool"
    )
    assert "raven-1" in observation.content
    assert final["messages"][-1].content == "Urban adaptation [raven-1]"
    assert final["messages"][-1].usage_metadata["input_tokens"] > 0


def test_chat_refuses_partial_index(tmp_path):
    with pytest.raises(ValueError, match="Build the index first"):
        open_index(tmp_path)


def test_ingestion_reuse_resume_and_settings_guard(tmp_path, monkeypatch):
    from lg_report.platform import rag_index

    monkeypatch.setattr(rag_index, "restore_archive", lambda *args: False)
    source = tmp_path / "source.parquet"
    empty = tmp_path / "empty.parquet"
    pq.write_table(
        pa.table({"text": [" = Test article = ", " ".join(["word"] * 441)]}), source
    )
    pq.write_table(pa.table({"text": pa.array([], type=pa.string())}), empty)
    tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0, "word": 1}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer_file = tmp_path / "tokenizer.json"
    tokenizer.save(str(tokenizer_file))

    def local_download(*args, filename, **kwargs):
        # Keep index lifecycle tests independent of external downloads/model caches.
        if filename == "tokenizer.json":
            return str(tokenizer_file)
        return str(source if "00000" in filename else empty)

    monkeypatch.setattr(rag_index, "hf_hub_download", local_download)
    monkeypatch.setattr(rag_index, "DefaultEmbeddingFunction", TestEmbedding)
    directory = tmp_path / "index"
    manifest = rag_index.build_index(directory, max_passages=3)
    assert manifest["passages"] == 3
    assert manifest["embedding_tokens_without_overlap"] == 441
    assert manifest["embedding_tokens_with_overlap"] == 505
    assert rag_index.open_index(directory).count() == 3
    assert rag_index.build_index(directory, max_passages=3) == manifest
    # Simulate interruption after database flush but before publishing completion.
    # All data already exists: reopening must recount and finish without embeddings.
    (directory / "manifest.json").unlink()
    monkeypatch.setattr(
        TestEmbedding,
        "__call__",
        lambda self, input: pytest.fail("Resume re-embedded persisted data"),
    )
    assert rag_index.build_index(directory, max_passages=3)["passages"] == 3
    with pytest.raises(ValueError, match="settings changed"):
        rag_index.build_index(directory, max_passages=4)
    collection = rag_index.open_index(directory)
    collection.delete(ids=["passage-00000000"])
    with pytest.raises(ValueError, match="count does not match"):
        rag_index.open_index(directory)
