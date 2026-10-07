"""Provenance models for reproducible index builds."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from resolverag.domain.models import ArtifactManifest, ChunkingStrategy


class IndexModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = Field(default=1, ge=1)


class ChunkingStatistics(IndexModel):
    strategy: ChunkingStrategy
    documents: int = Field(ge=0)
    chunks: int = Field(ge=0)
    chunks_per_document: float = Field(ge=0)
    minimum_tokens: int = Field(ge=0)
    mean_tokens: float = Field(ge=0)
    p50_tokens: int = Field(ge=0)
    p95_tokens: int = Field(ge=0)
    maximum_tokens: int = Field(ge=0)


class CollectionBuild(IndexModel):
    strategy: ChunkingStrategy
    collection_name: str = Field(min_length=1)
    vector_count: int = Field(ge=0)
    build_seconds: float = Field(ge=0)
    chunk_artifact: ArtifactManifest
    statistics: ChunkingStatistics


class IndexBuildManifest(IndexModel):
    generated_at: datetime
    scope: str = Field(min_length=1)
    document_limit: int | None = Field(default=None, gt=0)
    configuration_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    dataset_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    documents_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    embedding_provider: str = Field(min_length=1)
    embedding_model: str = Field(min_length=1)
    embedding_dimensions: int = Field(gt=0)
    qdrant_path: str = Field(min_length=1)
    collections: tuple[CollectionBuild, ...]
