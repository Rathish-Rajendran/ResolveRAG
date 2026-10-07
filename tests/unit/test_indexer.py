from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
import yaml
from qdrant_client import QdrantClient

from resolverag.data.integrity import sha256_file, sha256_text
from resolverag.domain.models import (
    ArtifactManifest,
    ChunkingStrategy,
    DatasetCounts,
    DatasetManifest,
    SourceDocument,
)
from resolverag.exceptions import ConfigurationError, DatasetValidationError, IndexBuildError
from resolverag.indexing.config import load_index_config
from resolverag.indexing.indexer import build_indexes


class FakeEmbeddings:
    def __init__(self, dimensions: int = 3) -> None:
        self.dimensions = dimensions
        self.embedded_texts: list[str] = []

    def embed_query(self, text: str) -> list[float]:
        self.embedded_texts.append(text)
        return [1.0] * self.dimensions

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.embedded_texts.extend(texts)
        return [[1.0, float(index + 1), 0.5] for index, _ in enumerate(texts)]


def _write_documents(tmp_path: Path) -> tuple[Path, DatasetManifest]:
    documents_path = tmp_path / "documents.jsonl"
    documents = [
        SourceDocument(
            document_id=f"doc-{index}.txt",
            text=(
                "Title: Example document. This is a support paragraph with several sentences. "
                "Restart the service and verify the configuration. "
            )
            * 4,
            source_path=f"corpus/doc-{index}.txt",
            text_sha256=sha256_text(f"document-{index}"),
        )
        for index in range(2)
    ]
    documents_path.write_text(
        "".join(document.model_dump_json() + "\n" for document in documents),
        encoding="utf-8",
    )
    artifact = ArtifactManifest(
        filename="documents.jsonl",
        byte_size=documents_path.stat().st_size,
        sha256=sha256_file(documents_path),
        records=2,
    )
    counts = DatasetCounts(
        source_question_rows=0,
        documents=2,
        queries=0,
        relevance_judgments=0,
        answerable_queries=0,
        unanswerable_queries=0,
        answerable_queries_without_reference_answer=0,
        development_queries=0,
        held_out_queries=0,
        duplicate_documents_removed=0,
    )
    manifest = DatasetManifest(
        dataset_name="test",
        repository_id="test/repository",
        repository_revision="revision",
        transformation_version="test-v1",
        generated_at=datetime.now(UTC),
        configuration_sha256="0" * 64,
        source_files=(),
        artifacts=(artifact,),
        counts=counts,
        library_versions={},
        code_revision=None,
        working_tree_dirty=None,
    )
    return documents_path, manifest


def _index_config(tmp_path: Path) -> tuple[Path, dict[str, Any]]:
    documents_path, dataset_manifest = _write_documents(tmp_path)
    dataset_manifest_path = tmp_path / "dataset-manifest.json"
    dataset_manifest_path.write_text(dataset_manifest.model_dump_json(), encoding="utf-8")
    config_data: dict[str, Any] = {
        "schema_version": 1,
        "inputs": {
            "documents_file": str(documents_path),
            "dataset_manifest_file": str(dataset_manifest_path),
        },
        "outputs": {"directory": str(tmp_path / "indexes")},
        "embedding": {
            "provider": "ollama",
            "model": "fake-embedding",
            "base_url": "http://127.0.0.1:11434",
            "expected_dimensions": 3,
            "document_prefix": "search_document: ",
            "query_prefix": "search_query: ",
        },
        "qdrant": {
            "path": str(tmp_path / "qdrant"),
            "collection_prefix": "test_techqa",
            "distance": "cosine",
            "embedding_batch_size": 2,
            "upload_batch_size": 2,
        },
        "tokenizer": {"encoding_name": "cl100k_base"},
        "strategies": {
            "fixed_256": {"kind": "token", "chunk_size": 20, "chunk_overlap": 2},
            "fixed_512": {"kind": "token", "chunk_size": 30, "chunk_overlap": 3},
            "recursive": {
                "kind": "recursive",
                "chunk_size": 100,
                "chunk_overlap": 10,
                "separators": ["\n\n", "\n", ". ", " ", ""],
            },
            "sentence_window": {
                "kind": "sentence_window",
                "window_size": 3,
                "sentence_overlap": 1,
                "max_sentence_characters": 100,
            },
            "parent_child": {
                "kind": "parent_child",
                "parent_chunk_size": 150,
                "parent_chunk_overlap": 10,
                "child_chunk_size": 60,
                "child_chunk_overlap": 5,
            },
        },
    }
    config_path = tmp_path / "index.yaml"
    config_path.write_text(yaml.safe_dump(config_data), encoding="utf-8")
    return config_path, config_data


def test_build_index_writes_chunks_and_exact_qdrant_count(tmp_path: Path) -> None:
    config_path, _ = _index_config(tmp_path)
    embeddings = FakeEmbeddings()
    client = QdrantClient(location=":memory:")
    try:
        result = build_indexes(
            config_path,
            strategies=[ChunkingStrategy.FIXED_256],
            document_limit=1,
            embeddings=embeddings,
            client=client,
        )

        collection = result.manifest.collections[0]
        assert collection.statistics.documents == 1
        assert collection.vector_count == collection.statistics.chunks
        assert collection.vector_count > 1
        assert client.count(collection.collection_name).count == collection.vector_count
        assert collection.chunk_artifact.records == collection.vector_count
        assert result.manifest.embedding_dimensions == 3
        assert result.manifest.scope == "smoke_1"
        assert result.manifest_path.is_file()
        assert result.statistics_path.is_file()
        assert all(text.startswith("search_document: ") for text in embeddings.embedded_texts)
    finally:
        client.close()


def test_build_index_rejects_non_positive_limit(tmp_path: Path) -> None:
    config_path, _ = _index_config(tmp_path)

    with pytest.raises(IndexBuildError, match="greater than zero"):
        build_indexes(config_path, document_limit=0, embeddings=FakeEmbeddings())


def test_build_index_rejects_document_checksum_drift(tmp_path: Path) -> None:
    config_path, config_data = _index_config(tmp_path)
    documents_path = Path(config_data["inputs"]["documents_file"])
    documents_path.write_text("changed\n", encoding="utf-8")

    with pytest.raises(DatasetValidationError, match="checksum mismatch"):
        build_indexes(config_path, embeddings=FakeEmbeddings())


def test_build_index_rejects_embedding_dimension_change(tmp_path: Path) -> None:
    config_path, _ = _index_config(tmp_path)

    with pytest.raises(IndexBuildError, match="returned 2 dimensions"):
        build_indexes(config_path, embeddings=FakeEmbeddings(dimensions=2))


def test_index_config_requires_all_benchmark_strategies(tmp_path: Path) -> None:
    config_path, config_data = _index_config(tmp_path)
    del config_data["strategies"]["parent_child"]
    config_path.write_text(yaml.safe_dump(config_data), encoding="utf-8")

    with pytest.raises(ConfigurationError, match="missing strategies"):
        load_index_config(config_path)
