"""Document-level information-retrieval metrics."""

import math
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievalMetrics:
    recall_at_1: float
    recall_at_5: float
    recall_at_10: float
    reciprocal_rank_at_10: float
    ndcg_at_10: float


def recall_at_k(
    retrieved_document_ids: Sequence[str],
    relevant_document_ids: set[str],
    k: int,
) -> float:
    if not relevant_document_ids:
        return 0.0
    retrieved = set(retrieved_document_ids[:k])
    return len(retrieved & relevant_document_ids) / len(relevant_document_ids)


def reciprocal_rank_at_k(
    retrieved_document_ids: Sequence[str],
    relevant_document_ids: set[str],
    k: int,
) -> float:
    for rank, document_id in enumerate(retrieved_document_ids[:k], start=1):
        if document_id in relevant_document_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(
    retrieved_document_ids: Sequence[str],
    relevant_document_ids: set[str],
    k: int,
) -> float:
    if not relevant_document_ids:
        return 0.0
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, document_id in enumerate(retrieved_document_ids[:k], start=1)
        if document_id in relevant_document_ids
    )
    ideal_hits = min(k, len(relevant_document_ids))
    ideal_dcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / ideal_dcg if ideal_dcg else 0.0


def evaluate_ranking(
    retrieved_document_ids: Sequence[str],
    relevant_document_ids: set[str],
) -> RetrievalMetrics:
    """Calculate the fixed metric suite used by the ResolveRAG leaderboard."""

    return RetrievalMetrics(
        recall_at_1=recall_at_k(retrieved_document_ids, relevant_document_ids, 1),
        recall_at_5=recall_at_k(retrieved_document_ids, relevant_document_ids, 5),
        recall_at_10=recall_at_k(retrieved_document_ids, relevant_document_ids, 10),
        reciprocal_rank_at_10=reciprocal_rank_at_k(
            retrieved_document_ids,
            relevant_document_ids,
            10,
        ),
        ndcg_at_10=ndcg_at_k(retrieved_document_ids, relevant_document_ids, 10),
    )
