"""Reciprocal-rank and normalized-score hybrid fusion."""

from collections.abc import Sequence
from dataclasses import dataclass

from resolverag.domain.models import (
    Chunk,
    RetrievalConfiguration,
    RetrievalResult,
)
from resolverag.retrieval.base import Retriever


@dataclass
class _FusedCandidate:
    chunk: Chunk
    score: float
    components: dict[str, float]


def _ranked_results(
    candidates: dict[str, _FusedCandidate],
    retriever: RetrievalConfiguration,
    top_k: int,
) -> tuple[RetrievalResult, ...]:
    ordered = sorted(candidates.values(), key=lambda candidate: candidate.score, reverse=True)
    return tuple(
        RetrievalResult(
            retriever=retriever,
            rank=rank,
            score=candidate.score,
            chunk=candidate.chunk,
            component_scores=candidate.components,
        )
        for rank, candidate in enumerate(ordered[:top_k], start=1)
    )


class ReciprocalRankFusionRetriever:
    """Fuse sparse and dense ranks without assuming comparable raw scores."""

    def __init__(
        self,
        sparse: Retriever,
        dense: Retriever,
        *,
        sparse_candidate_k: int,
        dense_candidate_k: int,
        rank_constant: int,
    ) -> None:
        self._sparse = sparse
        self._dense = dense
        self._sparse_candidate_k = sparse_candidate_k
        self._dense_candidate_k = dense_candidate_k
        self._rank_constant = rank_constant

    def retrieve(self, query: str, top_k: int) -> tuple[RetrievalResult, ...]:
        sparse_results = self._sparse.retrieve(query, self._sparse_candidate_k)
        dense_results = self._dense.retrieve(query, self._dense_candidate_k)
        candidates: dict[str, _FusedCandidate] = {}
        for source, results in (("bm25", sparse_results), ("dense", dense_results)):
            for result in results:
                contribution = 1.0 / (self._rank_constant + result.rank)
                candidate = candidates.setdefault(
                    result.chunk.chunk_id,
                    _FusedCandidate(chunk=result.chunk, score=0.0, components={}),
                )
                candidate.score += contribution
                candidate.components[f"{source}_raw"] = result.score
                candidate.components[f"{source}_rank"] = float(result.rank)
                candidate.components[f"{source}_rrf"] = contribution
        return _ranked_results(
            candidates,
            RetrievalConfiguration.HYBRID_RRF,
            top_k,
        )


def _normalize(results: Sequence[RetrievalResult]) -> dict[str, float]:
    if not results:
        return {}
    scores = [result.score for result in results]
    minimum = min(scores)
    maximum = max(scores)
    if maximum == minimum:
        return {result.chunk.chunk_id: 1.0 for result in results}
    scale = maximum - minimum
    return {result.chunk.chunk_id: (result.score - minimum) / scale for result in results}


class WeightedScoreFusionRetriever:
    """Fuse min-max normalized sparse and dense scores with explicit weights."""

    def __init__(
        self,
        sparse: Retriever,
        dense: Retriever,
        *,
        sparse_candidate_k: int,
        dense_candidate_k: int,
        dense_weight: float,
    ) -> None:
        self._sparse = sparse
        self._dense = dense
        self._sparse_candidate_k = sparse_candidate_k
        self._dense_candidate_k = dense_candidate_k
        self._dense_weight = dense_weight

    def retrieve(self, query: str, top_k: int) -> tuple[RetrievalResult, ...]:
        sparse_results = self._sparse.retrieve(query, self._sparse_candidate_k)
        dense_results = self._dense.retrieve(query, self._dense_candidate_k)
        normalized_sparse = _normalize(sparse_results)
        normalized_dense = _normalize(dense_results)
        sparse_weight = 1.0 - self._dense_weight
        candidates: dict[str, _FusedCandidate] = {}
        for source, results in (("bm25", sparse_results), ("dense", dense_results)):
            for result in results:
                candidates.setdefault(
                    result.chunk.chunk_id,
                    _FusedCandidate(chunk=result.chunk, score=0.0, components={}),
                ).components[f"{source}_raw"] = result.score
        for chunk_id, candidate in candidates.items():
            sparse_score = normalized_sparse.get(chunk_id, 0.0)
            dense_score = normalized_dense.get(chunk_id, 0.0)
            candidate.score = sparse_weight * sparse_score + self._dense_weight * dense_score
            candidate.components.update(
                {
                    "bm25_normalized": sparse_score,
                    "dense_normalized": dense_score,
                    "weighted_fusion": candidate.score,
                }
            )
        return _ranked_results(
            candidates,
            RetrievalConfiguration.HYBRID_WEIGHTED,
            top_k,
        )
