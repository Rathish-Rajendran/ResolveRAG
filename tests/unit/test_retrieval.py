from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, uuid5

import pytest
import yaml
from qdrant_client import QdrantClient, models

from resolverag.data.integrity import sha256_file, sha256_text
from resolverag.domain.models import (
    ArtifactManifest,
    Chunk,
    ChunkingStrategy,
    RetrievalConfiguration,
)
from resolverag.exceptions import RetrievalError
from resolverag.indexing.models import (
    ChunkingStatistics,
    CollectionBuild,
    IndexBuildManifest,
)
from resolverag.retrieval.runtime import RetrievalRuntime


class FakeQueryEmbedder:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def embed_query(self, text: str) -> list[float]:
        self.calls.append(text)
        return [1.0, 0.0, 0.0]


class FakeReranker:
    def score(self, query: str, passages: list[str]) -> list[float]:
        return [
            10.0 if "restart" in passage.lower() else float(index)
            for index, passage in enumerate(passages)
        ]


def _chunk(index: int, document_id: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=str(uuid5(NAMESPACE_URL, f"test:{index}")),
        document_id=document_id,
        source_path=f"corpus/{document_id}",
        strategy=ChunkingStrategy.FIXED_256,
        chunk_index=index,
        text=text,
        context_text=text,
        text_sha256=sha256_text(text),
        token_count=len(text.split()),
    )


def _strategies() -> dict[str, dict[str, Any]]:
    return {
        "fixed_256": {"kind": "token", "chunk_size": 256, "chunk_overlap": 32},
        "fixed_512": {"kind": "token", "chunk_size": 512, "chunk_overlap": 64},
        "recursive": {
            "kind": "recursive",
            "chunk_size": 1800,
            "chunk_overlap": 200,
            "separators": ["\n\n", "\n", ". ", " ", ""],
        },
        "sentence_window": {
            "kind": "sentence_window",
            "window_size": 5,
            "sentence_overlap": 1,
            "max_sentence_characters": 1800,
        },
        "parent_child": {
            "kind": "parent_child",
            "parent_chunk_size": 2400,
            "parent_chunk_overlap": 200,
            "child_chunk_size": 600,
            "child_chunk_overlap": 80,
        },
    }


def _runtime_files(tmp_path: Path, client: QdrantClient) -> tuple[Path, Path, list[Chunk]]:
    chunks = [
        _chunk(0, "restart.txt", "Restart the failed application service."),
        _chunk(1, "config.txt", "Verify the application configuration and environment."),
        _chunk(2, "network.txt", "Inspect network ports and firewall routing."),
        _chunk(3, "backup.txt", "Restore a database backup safely."),
    ]
    output_directory = tmp_path / "indexes" / "smoke_4"
    chunks_path = output_directory / "chunks" / "fixed_256.jsonl"
    chunks_path.parent.mkdir(parents=True)
    chunks_path.write_text(
        "".join(chunk.model_dump_json() + "\n" for chunk in chunks),
        encoding="utf-8",
    )
    index_config_data: dict[str, Any] = {
        "schema_version": 1,
        "inputs": {
            "documents_file": str(tmp_path / "documents.jsonl"),
            "dataset_manifest_file": str(tmp_path / "dataset-manifest.json"),
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
            "mode": "local",
            "path": str(tmp_path / "qdrant"),
            "collection_prefix": "test",
            "distance": "cosine",
            "embedding_batch_size": 2,
            "upload_batch_size": 2,
        },
        "tokenizer": {"encoding_name": "cl100k_base"},
        "strategies": _strategies(),
    }
    index_config_path = tmp_path / "index.yaml"
    index_config_path.write_text(yaml.safe_dump(index_config_data), encoding="utf-8")

    collection_name = "test_fixed_256_smoke_4"
    client.create_collection(
        collection_name,
        vectors_config=models.VectorParams(size=3, distance=models.Distance.COSINE),
    )
    vectors = ([1.0, 0.0, 0.0], [0.9, 0.1, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0])
    client.upload_points(
        collection_name,
        [
            models.PointStruct(
                id=chunk.chunk_id,
                vector=vector,
                payload=chunk.model_dump(mode="json"),
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ],
        wait=True,
    )
    artifact = ArtifactManifest(
        filename="chunks/fixed_256.jsonl",
        byte_size=chunks_path.stat().st_size,
        sha256=sha256_file(chunks_path),
        records=len(chunks),
    )
    statistics = ChunkingStatistics(
        strategy=ChunkingStrategy.FIXED_256,
        documents=4,
        chunks=4,
        chunks_per_document=1.0,
        minimum_tokens=5,
        mean_tokens=5.0,
        p50_tokens=5,
        p95_tokens=5,
        maximum_tokens=5,
    )
    collection = CollectionBuild(
        strategy=ChunkingStrategy.FIXED_256,
        collection_name=collection_name,
        vector_count=4,
        build_seconds=0.1,
        chunk_artifact=artifact,
        statistics=statistics,
    )
    index_manifest = IndexBuildManifest(
        generated_at=datetime.now(UTC),
        scope="smoke_4",
        document_limit=4,
        configuration_sha256=sha256_file(index_config_path),
        dataset_manifest_sha256="1" * 64,
        documents_sha256="2" * 64,
        embedding_provider="ollama",
        embedding_model="fake-embedding",
        embedding_dimensions=3,
        qdrant_mode="local",
        qdrant_location=str(tmp_path / "qdrant"),
        collections=(collection,),
    )
    index_manifest_path = output_directory / "manifest.json"
    index_manifest_path.write_text(index_manifest.model_dump_json(), encoding="utf-8")

    retrieval_config_data = {
        "schema_version": 1,
        "inputs": {
            "index_config_file": str(index_config_path),
            "index_manifest_file": str(index_manifest_path),
        },
        "retrieval": {
            "default_top_k": 2,
            "dense_candidate_k": 4,
            "sparse_candidate_k": 4,
            "hybrid_candidate_k": 4,
            "mmr_lambda": 0.65,
            "reciprocal_rank_constant": 60,
            "dense_weight": 0.6,
        },
        "reranker": {"model": "fake-reranker", "batch_size": 2},
    }
    retrieval_config_path = tmp_path / "retrieval.yaml"
    retrieval_config_path.write_text(yaml.safe_dump(retrieval_config_data), encoding="utf-8")
    return retrieval_config_path, index_manifest_path, chunks


def test_all_six_retrieval_configurations_return_traceable_results(tmp_path: Path) -> None:
    client = QdrantClient(location=":memory:")
    embedder = FakeQueryEmbedder()
    try:
        config_path, manifest_path, _ = _runtime_files(tmp_path, client)
        with RetrievalRuntime(
            config_path,
            ChunkingStrategy.FIXED_256,
            index_manifest_path=manifest_path,
            embedder=embedder,
            reranker=FakeReranker(),
            client=client,
        ) as runtime:
            for configuration in RetrievalConfiguration:
                results = runtime.search(
                    "restart application service",
                    configuration,
                    top_k=2,
                )
                assert len(results) == 2
                assert [result.rank for result in results] == [1, 2]
                assert all(result.retriever is configuration for result in results)
                assert all(result.chunk.document_id for result in results)
                assert all(result.component_scores for result in results)

            assert (
                runtime.search(
                    "restart application service",
                    RetrievalConfiguration.BM25,
                    top_k=1,
                )[0].chunk.document_id
                == "restart.txt"
            )
            assert (
                runtime.search(
                    "restart application service",
                    RetrievalConfiguration.HYBRID_RERANK,
                    top_k=1,
                )[0].chunk.document_id
                == "restart.txt"
            )
        assert embedder.calls == ["search_query: restart application service"]
    finally:
        client.close()


def test_runtime_rejects_empty_query_and_excessive_top_k(tmp_path: Path) -> None:
    client = QdrantClient(location=":memory:")
    try:
        config_path, manifest_path, _ = _runtime_files(tmp_path, client)
        runtime = RetrievalRuntime(
            config_path,
            ChunkingStrategy.FIXED_256,
            index_manifest_path=manifest_path,
            embedder=FakeQueryEmbedder(),
            reranker=FakeReranker(),
            client=client,
        )
        with pytest.raises(RetrievalError, match="must not be empty"):
            runtime.search(" ", RetrievalConfiguration.BM25)
        with pytest.raises(RetrievalError, match="cannot exceed"):
            runtime.search("query", RetrievalConfiguration.BM25, top_k=5)
    finally:
        client.close()


def test_runtime_detects_chunk_artifact_drift(tmp_path: Path) -> None:
    client = QdrantClient(location=":memory:")
    try:
        config_path, manifest_path, chunks = _runtime_files(tmp_path, client)
        chunks_path = manifest_path.parent / "chunks" / "fixed_256.jsonl"
        chunks_path.write_text(chunks[0].model_dump_json() + "\n", encoding="utf-8")
        runtime = RetrievalRuntime(
            config_path,
            ChunkingStrategy.FIXED_256,
            index_manifest_path=manifest_path,
            client=client,
        )
        with pytest.raises(RetrievalError, match="checksum mismatch"):
            runtime.search("restart", RetrievalConfiguration.BM25)
    finally:
        client.close()
