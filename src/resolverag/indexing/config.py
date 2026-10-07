"""Typed configuration for chunking and local vector indexes."""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from resolverag.domain.models import ChunkingStrategy
from resolverag.exceptions import ConfigurationError


class StrictIndexConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class IndexInputs(StrictIndexConfigModel):
    documents_file: Path
    dataset_manifest_file: Path


class IndexOutputs(StrictIndexConfigModel):
    directory: Path


class EmbeddingSettings(StrictIndexConfigModel):
    provider: Literal["ollama"]
    model: str = Field(min_length=1)
    base_url: str = Field(min_length=1)
    expected_dimensions: int = Field(gt=0)
    document_prefix: str
    query_prefix: str


class QdrantSettings(StrictIndexConfigModel):
    mode: Literal["local", "server"]
    path: Path | None = None
    url: str | None = Field(default=None, min_length=1)
    prefer_grpc: bool = True
    timeout_seconds: int = Field(default=120, gt=0)
    collection_prefix: str = Field(min_length=1, pattern=r"^[a-z0-9_]+$")
    distance: Literal["cosine"]
    embedding_batch_size: int = Field(gt=0)
    upload_batch_size: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_location(self) -> "QdrantSettings":
        """Require exactly the location used by the selected backend."""

        if self.mode == "local":
            if self.path is None or self.url is not None:
                raise ValueError("local Qdrant mode requires path and forbids url")
        elif self.url is None or self.path is not None:
            raise ValueError("server Qdrant mode requires url and forbids path")
        return self

    @property
    def location(self) -> str:
        """Return a stable location string for clients and provenance."""

        if self.mode == "local":
            if self.path is None:  # pragma: no cover - enforced by validation
                raise ValueError("Local Qdrant path is missing")
            return str(self.path)
        if self.url is None:  # pragma: no cover - enforced by validation
            raise ValueError("Qdrant server URL is missing")
        return self.url


class TokenizerSettings(StrictIndexConfigModel):
    encoding_name: str = Field(min_length=1)


class StrategySettings(StrictIndexConfigModel):
    kind: Literal["token", "recursive", "sentence_window", "parent_child"]
    chunk_size: int | None = Field(default=None, gt=0)
    chunk_overlap: int | None = Field(default=None, ge=0)
    separators: list[str] | None = None
    window_size: int | None = Field(default=None, gt=0)
    sentence_overlap: int | None = Field(default=None, ge=0)
    max_sentence_characters: int | None = Field(default=None, gt=0)
    parent_chunk_size: int | None = Field(default=None, gt=0)
    parent_chunk_overlap: int | None = Field(default=None, ge=0)
    child_chunk_size: int | None = Field(default=None, gt=0)
    child_chunk_overlap: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_parameters(self) -> "StrategySettings":
        """Require the parameters used by the selected strategy."""

        if self.kind in {"token", "recursive"}:
            if self.chunk_size is None or self.chunk_overlap is None:
                raise ValueError(f"{self.kind} requires chunk_size and chunk_overlap")
            if self.chunk_overlap >= self.chunk_size:
                raise ValueError("chunk_overlap must be smaller than chunk_size")
        if self.kind == "recursive" and not self.separators:
            raise ValueError("recursive requires at least one separator")
        if self.kind == "sentence_window":
            if (
                self.window_size is None
                or self.sentence_overlap is None
                or self.max_sentence_characters is None
            ):
                raise ValueError(
                    "sentence_window requires window_size, sentence_overlap, and "
                    "max_sentence_characters"
                )
            if self.sentence_overlap >= self.window_size:
                raise ValueError("sentence_overlap must be smaller than window_size")
        if self.kind == "parent_child":
            parent_size = self.parent_chunk_size
            parent_overlap = self.parent_chunk_overlap
            child_size = self.child_chunk_size
            child_overlap = self.child_chunk_overlap
            if (
                parent_size is None
                or parent_overlap is None
                or child_size is None
                or child_overlap is None
            ):
                raise ValueError("parent_child requires all parent and child size settings")
            if parent_overlap >= parent_size:
                raise ValueError("parent_chunk_overlap must be smaller than parent_chunk_size")
            if child_overlap >= child_size:
                raise ValueError("child_chunk_overlap must be smaller than child_chunk_size")
        return self


class IndexConfig(StrictIndexConfigModel):
    schema_version: int = Field(ge=1)
    inputs: IndexInputs
    outputs: IndexOutputs
    embedding: EmbeddingSettings
    qdrant: QdrantSettings
    tokenizer: TokenizerSettings
    strategies: dict[ChunkingStrategy, StrategySettings]


def load_index_config(path: Path) -> IndexConfig:
    """Load one complete chunking and index configuration."""

    if not path.is_file():
        raise ConfigurationError(f"Index configuration does not exist: {path}")
    try:
        raw_config: object = yaml.safe_load(path.read_text(encoding="utf-8"))
        config = IndexConfig.model_validate(raw_config)
    except (OSError, yaml.YAMLError, ValidationError) as error:
        raise ConfigurationError(f"Invalid index configuration {path}: {error}") from error
    missing = set(ChunkingStrategy) - set(config.strategies)
    if missing:
        missing_names = ", ".join(sorted(strategy.value for strategy in missing))
        raise ConfigurationError(f"Index configuration is missing strategies: {missing_names}")
    return config
