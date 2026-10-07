"""Canonical data models shared by ingestion, retrieval, and evaluation."""

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class CanonicalModel(BaseModel):
    """Strict base model for versioned records owned by ResolveRAG."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = Field(default=1, ge=1)


class LogicalSplit(StrEnum):
    """Permitted uses of benchmark questions."""

    DEVELOPMENT = "development"
    HELD_OUT = "held_out"


class ChunkingStrategy(StrEnum):
    """Chunking alternatives included in the retrieval benchmark."""

    FIXED_256 = "fixed_256"
    FIXED_512 = "fixed_512"
    RECURSIVE = "recursive"
    SENTENCE_WINDOW = "sentence_window"
    PARENT_CHILD = "parent_child"


class RetrievalConfiguration(StrEnum):
    """Retrieval alternatives included in the benchmark matrix."""

    BM25 = "bm25"
    DENSE = "dense"
    DENSE_MMR = "dense_mmr"
    HYBRID_RRF = "hybrid_rrf"
    HYBRID_WEIGHTED = "hybrid_weighted"
    HYBRID_RERANK = "hybrid_rerank"


class SourceDocument(CanonicalModel):
    """A normalized technical document that may be indexed."""

    document_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    source_path: str = Field(min_length=1)
    text_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class Chunk(CanonicalModel):
    """An indexable passage with traceability to its source and optional parent."""

    chunk_id: str = Field(pattern=r"^[0-9a-f-]{36}$")
    document_id: str = Field(min_length=1)
    source_path: str = Field(min_length=1)
    strategy: ChunkingStrategy
    chunk_index: int = Field(ge=0)
    text: str = Field(min_length=1)
    context_text: str = Field(min_length=1)
    parent_chunk_id: str | None = Field(default=None, pattern=r"^[0-9a-f-]{36}$")
    text_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    token_count: int = Field(ge=1)


class RetrievalResult(CanonicalModel):
    """One ranked chunk with transparent component scores."""

    retriever: RetrievalConfiguration
    rank: int = Field(ge=1)
    score: float
    chunk: Chunk
    component_scores: dict[str, float]


class EvaluationQuery(CanonicalModel):
    """A benchmark question kept strictly outside the retrieval index."""

    query_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    reference_answer: str | None
    is_answerable: bool
    split: LogicalSplit


class RelevanceJudgment(CanonicalModel):
    """A binary relevance label connecting a query to a source document."""

    query_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    relevance: int = Field(default=1, ge=1)


class SourceFileManifest(CanonicalModel):
    """Integrity information for one immutable downloaded source file."""

    filename: str = Field(min_length=1)
    byte_size: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ArtifactManifest(SourceFileManifest):
    """Integrity and record-count information for a generated artifact."""

    records: int = Field(ge=0)


class DatasetCounts(CanonicalModel):
    """Observed record counts for a prepared dataset."""

    source_question_rows: int = Field(ge=0)
    documents: int = Field(ge=0)
    queries: int = Field(ge=0)
    relevance_judgments: int = Field(ge=0)
    answerable_queries: int = Field(ge=0)
    unanswerable_queries: int = Field(ge=0)
    answerable_queries_without_reference_answer: int = Field(ge=0)
    development_queries: int = Field(ge=0)
    held_out_queries: int = Field(ge=0)
    duplicate_documents_removed: int = Field(ge=0)


class DatasetManifest(CanonicalModel):
    """Provenance needed to reproduce one canonical dataset build."""

    dataset_name: str = Field(min_length=1)
    repository_id: str = Field(min_length=1)
    repository_revision: str = Field(min_length=1)
    transformation_version: str = Field(min_length=1)
    generated_at: datetime
    configuration_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_files: tuple[SourceFileManifest, ...]
    artifacts: tuple[ArtifactManifest, ...]
    counts: DatasetCounts
    library_versions: dict[str, str]
    code_revision: str | None
    working_tree_dirty: bool | None
