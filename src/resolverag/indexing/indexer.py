"""Streaming chunk generation, Ollama embedding, and Qdrant persistence."""

import json
import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Protocol, cast

from langchain_ollama import OllamaEmbeddings
from pydantic import ValidationError
from qdrant_client import QdrantClient, models

from resolverag.data.integrity import sha256_file
from resolverag.data.serialization import atomic_text_writer, write_text
from resolverag.domain.models import (
    ArtifactManifest,
    Chunk,
    ChunkingStrategy,
    DatasetManifest,
    SourceDocument,
)
from resolverag.exceptions import DatasetValidationError, IndexBuildError
from resolverag.indexing.chunking import build_chunker
from resolverag.indexing.config import IndexConfig, load_index_config
from resolverag.indexing.models import (
    ChunkingStatistics,
    CollectionBuild,
    IndexBuildManifest,
)


class EmbeddingProvider(Protocol):
    """Minimal embedding interface used by the index builder."""

    def embed_query(self, text: str) -> list[float]: ...

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...


@dataclass(frozen=True)
class IndexBuildResult:
    """Paths and provenance returned by a completed build."""

    output_directory: Path
    manifest_path: Path
    statistics_path: Path
    manifest: IndexBuildManifest


def _load_dataset_manifest(path: Path) -> DatasetManifest:
    try:
        return DatasetManifest.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValidationError) as error:
        raise DatasetValidationError(f"Invalid dataset manifest {path}: {error}") from error


def _validate_documents(config: IndexConfig, manifest: DatasetManifest) -> None:
    path = config.inputs.documents_file
    if not path.is_file():
        raise DatasetValidationError(
            f"Canonical documents are missing; run dataset prepare first: {path}"
        )
    expected = next(
        (artifact.sha256 for artifact in manifest.artifacts if artifact.filename == path.name),
        None,
    )
    if expected is None:
        raise DatasetValidationError(f"Dataset manifest does not describe {path.name}")
    observed = sha256_file(path)
    if observed != expected:
        raise DatasetValidationError(
            f"Canonical document checksum mismatch: expected {expected}, observed {observed}"
        )


def _iter_documents(path: Path, limit: int | None) -> Iterator[SourceDocument]:
    try:
        with path.open(encoding="utf-8") as source:
            for index, line in enumerate(source):
                if limit is not None and index >= limit:
                    break
                try:
                    yield SourceDocument.model_validate_json(line)
                except ValidationError as error:
                    raise DatasetValidationError(
                        f"Invalid canonical document at line {index + 1}: {error}"
                    ) from error
    except OSError as error:
        raise DatasetValidationError(
            f"Unable to read canonical documents {path}: {error}"
        ) from error


def _percentile(values: Sequence[int], percentile: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    rank = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[rank]


def _statistics(
    strategy: ChunkingStrategy,
    document_count: int,
    token_counts: Sequence[int],
) -> ChunkingStatistics:
    chunk_count = len(token_counts)
    return ChunkingStatistics(
        strategy=strategy,
        documents=document_count,
        chunks=chunk_count,
        chunks_per_document=chunk_count / document_count if document_count else 0.0,
        minimum_tokens=min(token_counts, default=0),
        mean_tokens=sum(token_counts) / chunk_count if chunk_count else 0.0,
        p50_tokens=_percentile(token_counts, 0.50),
        p95_tokens=_percentile(token_counts, 0.95),
        maximum_tokens=max(token_counts, default=0),
    )


def _payload(chunk: Chunk) -> dict[str, object]:
    return cast(dict[str, object], chunk.model_dump(mode="json", exclude_none=False))


def _upload_batch(
    client: QdrantClient,
    collection_name: str,
    chunks: Sequence[Chunk],
    embeddings: EmbeddingProvider,
    config: IndexConfig,
) -> None:
    texts = [f"{config.embedding.document_prefix}{chunk.text}" for chunk in chunks]
    vectors = embeddings.embed_documents(texts)
    if len(vectors) != len(chunks):
        raise IndexBuildError(
            f"Embedding provider returned {len(vectors)} vectors for {len(chunks)} chunks"
        )
    points: list[models.PointStruct] = []
    for chunk, vector in zip(chunks, vectors, strict=True):
        if len(vector) != config.embedding.expected_dimensions:
            raise IndexBuildError(
                f"Embedding dimension changed from {config.embedding.expected_dimensions} "
                f"to {len(vector)}"
            )
        points.append(models.PointStruct(id=chunk.chunk_id, vector=vector, payload=_payload(chunk)))
    client.upload_points(
        collection_name=collection_name,
        points=points,
        batch_size=config.qdrant.upload_batch_size,
        wait=True,
    )


def _collection_name(
    config: IndexConfig,
    strategy: ChunkingStrategy,
    limit: int | None,
) -> str:
    suffix = strategy.value if limit is None else f"{strategy.value}_smoke_{limit}"
    return f"{config.qdrant.collection_prefix}_{suffix}"


def _prepare_collection(
    client: QdrantClient,
    collection_name: str,
    dimensions: int,
    *,
    recreate: bool,
) -> None:
    exists = client.collection_exists(collection_name)
    if exists and not recreate:
        raise IndexBuildError(
            f"Collection {collection_name!r} already exists; use --recreate to replace it"
        )
    if exists:
        client.delete_collection(collection_name)
    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=dimensions, distance=models.Distance.COSINE),
    )


def _build_collection(
    client: QdrantClient,
    embeddings: EmbeddingProvider,
    config: IndexConfig,
    strategy: ChunkingStrategy,
    output_directory: Path,
    dimensions: int,
    limit: int | None,
    *,
    recreate: bool,
) -> CollectionBuild:
    collection_name = _collection_name(config, strategy, limit)
    _prepare_collection(client, collection_name, dimensions, recreate=recreate)
    chunker = build_chunker(
        strategy,
        config.strategies[strategy],
        encoding_name=config.tokenizer.encoding_name,
    )
    chunks_path = output_directory / "chunks" / f"{strategy.value}.jsonl"
    batch: list[Chunk] = []
    token_counts: list[int] = []
    document_count = 0
    started = perf_counter()

    try:
        with atomic_text_writer(chunks_path) as destination:
            for document in _iter_documents(config.inputs.documents_file, limit):
                document_count += 1
                chunks = chunker.split(document)
                if not chunks:
                    raise IndexBuildError(
                        f"Strategy {strategy.value!r} produced no chunks for "
                        f"document {document.document_id!r}"
                    )
                for chunk in chunks:
                    destination.write(chunk.model_dump_json(exclude_none=False))
                    destination.write("\n")
                    token_counts.append(chunk.token_count)
                    batch.append(chunk)
                    if len(batch) >= config.qdrant.embedding_batch_size:
                        _upload_batch(client, collection_name, batch, embeddings, config)
                        batch.clear()
            if batch:
                _upload_batch(client, collection_name, batch, embeddings, config)
                batch.clear()
    except Exception:
        client.delete_collection(collection_name)
        raise

    if document_count == 0:
        client.delete_collection(collection_name)
        raise IndexBuildError("No canonical documents were available for indexing")
    vector_count = client.count(collection_name=collection_name, exact=True).count
    if vector_count != len(token_counts):
        client.delete_collection(collection_name)
        raise IndexBuildError(
            f"Collection {collection_name!r} contains {vector_count} vectors; "
            f"expected {len(token_counts)}"
        )
    chunk_artifact = ArtifactManifest(
        filename=str(chunks_path.relative_to(output_directory)),
        byte_size=chunks_path.stat().st_size,
        sha256=sha256_file(chunks_path),
        records=len(token_counts),
    )
    return CollectionBuild(
        strategy=strategy,
        collection_name=collection_name,
        vector_count=vector_count,
        build_seconds=perf_counter() - started,
        chunk_artifact=chunk_artifact,
        statistics=_statistics(strategy, document_count, token_counts),
    )


def _ollama_embeddings(config: IndexConfig) -> EmbeddingProvider:
    return OllamaEmbeddings(
        model=config.embedding.model,
        base_url=config.embedding.base_url,
        validate_model_on_init=True,
    )


def build_indexes(
    config_path: Path,
    *,
    strategies: Sequence[ChunkingStrategy] | None = None,
    document_limit: int | None = None,
    recreate: bool = True,
    embeddings: EmbeddingProvider | None = None,
    client: QdrantClient | None = None,
) -> IndexBuildResult:
    """Build configured Qdrant collections and chunk artifacts."""

    if document_limit is not None and document_limit <= 0:
        raise IndexBuildError("document_limit must be greater than zero")
    config = load_index_config(config_path)
    selected = tuple(strategies or tuple(ChunkingStrategy))
    if not selected:
        raise IndexBuildError("At least one chunking strategy must be selected")
    unknown = set(selected) - set(config.strategies)
    if unknown:
        names = ", ".join(sorted(item.value for item in unknown))
        raise IndexBuildError(f"Unconfigured chunking strategies: {names}")

    dataset_manifest = _load_dataset_manifest(config.inputs.dataset_manifest_file)
    _validate_documents(config, dataset_manifest)
    embedding_provider = embeddings or _ollama_embeddings(config)
    probe = embedding_provider.embed_query(
        f"{config.embedding.document_prefix}dimension verification"
    )
    dimensions = len(probe)
    if dimensions != config.embedding.expected_dimensions:
        raise IndexBuildError(
            f"Embedding model returned {dimensions} dimensions; "
            f"expected {config.embedding.expected_dimensions}"
        )

    scope = "full" if document_limit is None else f"smoke_{document_limit}"
    output_directory = config.outputs.directory / scope
    output_directory.mkdir(parents=True, exist_ok=True)
    owns_client = client is None
    qdrant_client = client or QdrantClient(path=str(config.qdrant.path))
    try:
        collections = tuple(
            _build_collection(
                qdrant_client,
                embedding_provider,
                config,
                strategy,
                output_directory,
                dimensions,
                document_limit,
                recreate=recreate,
            )
            for strategy in selected
        )
    finally:
        if owns_client:
            qdrant_client.close()

    manifest = IndexBuildManifest(
        generated_at=datetime.now(UTC),
        scope=scope,
        document_limit=document_limit,
        configuration_sha256=sha256_file(config_path),
        dataset_manifest_sha256=sha256_file(config.inputs.dataset_manifest_file),
        documents_sha256=sha256_file(config.inputs.documents_file),
        embedding_provider=config.embedding.provider,
        embedding_model=config.embedding.model,
        embedding_dimensions=dimensions,
        qdrant_path=str(config.qdrant.path),
        collections=collections,
    )
    manifest_path = output_directory / "manifest.json"
    statistics_path = output_directory / "chunking_statistics.json"
    write_text(manifest_path, manifest.model_dump_json(indent=2) + "\n")
    write_text(
        statistics_path,
        json.dumps(
            [collection.statistics.model_dump(mode="json") for collection in collections],
            indent=2,
        )
        + "\n",
    )
    return IndexBuildResult(
        output_directory=output_directory,
        manifest_path=manifest_path,
        statistics_path=statistics_path,
        manifest=manifest,
    )
