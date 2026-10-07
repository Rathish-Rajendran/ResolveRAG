"""Dense cosine and maximal-marginal-relevance retrieval."""

from typing import cast

import numpy as np
import numpy.typing as npt
from pydantic import ValidationError
from qdrant_client import QdrantClient, models

from resolverag.domain.models import (
    Chunk,
    RetrievalConfiguration,
    RetrievalResult,
)
from resolverag.exceptions import RetrievalError
from resolverag.retrieval.base import QueryEmbedder


def _cosine(left: npt.NDArray[np.float64], right: npt.NDArray[np.float64]) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator == 0:
        return 0.0
    return float(np.dot(left, right) / denominator)


class DenseRetriever:
    """Query one Qdrant collection with dense cosine similarity or MMR."""

    def __init__(
        self,
        client: QdrantClient,
        collection_name: str,
        embedder: QueryEmbedder,
        *,
        query_prefix: str,
        candidate_k: int,
        mmr_lambda: float | None = None,
    ) -> None:
        self._client = client
        self._collection_name = collection_name
        self._embedder = embedder
        self._query_prefix = query_prefix
        self._candidate_k = candidate_k
        self._mmr_lambda = mmr_lambda

    def _query(self, query: str, limit: int) -> tuple[list[float], list[models.ScoredPoint]]:
        vector = self._embedder.embed_query(f"{self._query_prefix}{query}")
        response = self._client.query_points(
            collection_name=self._collection_name,
            query=vector,
            limit=limit,
            with_payload=True,
            with_vectors=self._mmr_lambda is not None,
        )
        return vector, response.points

    def retrieve(self, query: str, top_k: int) -> tuple[RetrievalResult, ...]:
        if top_k <= 0:
            raise RetrievalError("top_k must be greater than zero")
        candidate_count = max(top_k, self._candidate_k if self._mmr_lambda is not None else top_k)
        query_vector, points = self._query(query, candidate_count)

        candidates: list[tuple[RetrievalResult, npt.NDArray[np.float64] | None]] = []
        for rank, point in enumerate(points, start=1):
            payload = point.payload
            if not isinstance(payload, dict):
                raise RetrievalError("Qdrant result is missing its canonical chunk payload")
            try:
                chunk = Chunk.model_validate(payload)
            except ValidationError as error:
                raise RetrievalError(
                    f"Invalid chunk payload returned by Qdrant: {error}"
                ) from error
            score = float(point.score)
            raw_vector = point.vector
            candidate_vector: npt.NDArray[np.float64] | None = None
            if self._mmr_lambda is not None:
                if not isinstance(raw_vector, list):
                    raise RetrievalError("MMR requires Qdrant to return dense candidate vectors")
                candidate_vector = np.asarray(raw_vector, dtype=np.float64)
            candidates.append(
                (
                    RetrievalResult(
                        retriever=RetrievalConfiguration.DENSE,
                        rank=rank,
                        score=score,
                        chunk=chunk,
                        component_scores={"dense": score},
                    ),
                    candidate_vector,
                )
            )

        if self._mmr_lambda is None:
            return tuple(result for result, _ in candidates[:top_k])
        return self._mmr(query_vector, candidates, top_k)

    def _mmr(
        self,
        query_vector: list[float],
        candidates: list[tuple[RetrievalResult, npt.NDArray[np.float64] | None]],
        top_k: int,
    ) -> tuple[RetrievalResult, ...]:
        query_array = np.asarray(query_vector, dtype=np.float64)
        remaining = list(candidates)
        selected: list[tuple[RetrievalResult, npt.NDArray[np.float64]]] = []
        output: list[RetrievalResult] = []
        relevance_weight = cast(float, self._mmr_lambda)
        while remaining and len(output) < top_k:
            best_index = 0
            best_score = float("-inf")
            for index, (_result, vector) in enumerate(remaining):
                if vector is None:
                    raise RetrievalError("MMR candidate is missing its vector")
                relevance = _cosine(query_array, vector)
                redundancy = max(
                    (_cosine(vector, selected_vector) for _, selected_vector in selected),
                    default=0.0,
                )
                mmr_score = relevance_weight * relevance - (1.0 - relevance_weight) * redundancy
                if mmr_score > best_score:
                    best_index = index
                    best_score = mmr_score
            result, vector = remaining.pop(best_index)
            if vector is None:
                raise RetrievalError("MMR candidate is missing its vector")
            selected.append((result, vector))
            components = dict(result.component_scores)
            components["mmr"] = best_score
            output.append(
                result.model_copy(
                    update={
                        "retriever": RetrievalConfiguration.DENSE_MMR,
                        "rank": len(output) + 1,
                        "score": best_score,
                        "component_scores": components,
                    }
                )
            )
        return tuple(output)
