"""Canonical benchmark records and aggregate leaderboard rows."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from resolverag.domain.models import (
    ChunkingStrategy,
    LogicalSplit,
    RetrievalConfiguration,
)


class EvaluationModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = Field(default=1, ge=1)


class QueryEvaluation(EvaluationModel):
    query_id: str = Field(min_length=1)
    strategy: ChunkingStrategy
    retriever: RetrievalConfiguration
    latency_ms: float = Field(ge=0)
    retrieved_document_ids: tuple[str, ...]
    relevant_document_ids: tuple[str, ...]
    recall_at_1: float = Field(ge=0, le=1)
    recall_at_5: float = Field(ge=0, le=1)
    recall_at_10: float = Field(ge=0, le=1)
    reciprocal_rank_at_10: float = Field(ge=0, le=1)
    ndcg_at_10: float = Field(ge=0, le=1)
    error: str | None = None


class LeaderboardEntry(EvaluationModel):
    strategy: ChunkingStrategy
    retriever: RetrievalConfiguration
    evaluated_queries: int = Field(ge=0)
    failures: int = Field(ge=0)
    recall_at_1: float = Field(ge=0, le=1)
    recall_at_5: float = Field(ge=0, le=1)
    recall_at_10: float = Field(ge=0, le=1)
    mrr_at_10: float = Field(ge=0, le=1)
    ndcg_at_10: float = Field(ge=0, le=1)
    p50_latency_ms: float = Field(ge=0)
    p95_latency_ms: float = Field(ge=0)


class BenchmarkManifest(EvaluationModel):
    run_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    status: str
    official: bool
    started_at: datetime
    completed_at: datetime | None
    index_scope: str = Field(min_length=1)
    split: LogicalSplit
    sample_query_ids: tuple[str, ...]
    strategies: tuple[ChunkingStrategy, ...]
    retrievers: tuple[RetrievalConfiguration, ...]
    evaluation_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    retrieval_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    index_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
